"""LogicAgent — pass B (decision + propose-only actions).

Aggregates the three analysts' `AgentResponse`s on the `logic` topic. Once all
expected sources have replied, runs a single LLM call with the write tools'
schemas visible. The LLM returns either:
  * a plain text explanation (no action warranted), or
  * one or more `FunctionCall` objects naming actions the LLM wants to take.

In this pass the tool calls are NOT executed — they are rendered as "proposed
actions" via `console.final_answer_box`. Flipping to real execution is a small
change in `_decide_and_propose`: replace the rendering step with the same
tool-loop that `AIAgent.handle_task` runs in `base.py`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)

from src.common import console
from src.mcp.client import MCPClient, MCPToolWrapper
from src.primitives.contracts import AgentResponse


LOGIC_TOOLS = [
    "update_task_attribute",
    "update_task_field",
    "link_evidence",
]


LOGIC_SYSTEM_PROMPT = """You are the LogicAgent.

You receive findings from three specialist analysts about a single event:
- ProcessStateAnalyst — task statuses and dependency edges.
- EvidenceAnalyst — communication trail, prior decisions, and reasons a task is blocked.
- ContextResearchAgent — broader graph context and historical precedents.

Your job is to decide what action (if any) the system should take in response
to the event. You have three write tools available:
- update_task_attribute — change a non-core attribute of a task.
- update_task_field — change a core field of a task (e.g. status).
- link_evidence — attach a new evidence record to a task.

Decide based strictly on the findings:
- If one or more of these actions are warranted, call the corresponding tool(s)
  with concrete arguments.
- If no action is warranted, return a short text explanation of why.

Cite task_ids, evidence_ids, and decision_ids verbatim from the findings.
Do not invent identifiers."""


def _extract_text(items: Any) -> str:
    """Pull user-visible text out of an AgentResponse.context payload."""
    if isinstance(items, str):
        return items
    if not isinstance(items, list):
        return str(items)
    pieces: List[str] = []
    for item in items:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(item, str):
            pieces.append(item)
    return "\n\n".join(pieces) if pieces else str(items)


def _render_proposed_actions(content: Any) -> str:
    """Turn an LLM response into a human-readable proposal string.

    The LLM either returns plain text (no action) or a list of FunctionCalls.
    Tool calls are NOT executed; they are formatted as proposals.
    """
    if isinstance(content, str):
        return f"No action proposed.\n\n{content}"

    if isinstance(content, list):
        lines: List[str] = ["Proposed actions (not executed):"]
        for call in content:
            if isinstance(call, FunctionCall):
                try:
                    args = json.loads(call.arguments)
                    args_render = ", ".join(f"{k}={v!r}" for k, v in args.items())
                except (json.JSONDecodeError, TypeError):
                    args_render = call.arguments
                lines.append(f"  - {call.name}({args_render})")
            else:
                lines.append(f"  - (unrecognized) {call!r}")
        return "\n".join(lines)

    return f"(unrecognized response shape: {content!r})"


class LogicAgent(RoutedAgent):
    def __init__(
        self,
        logic_topic_type: str,
        expected_sources: Set[str],
        model_client: ChatCompletionClient,
        tools: List[MCPToolWrapper],
    ) -> None:
        super().__init__(logic_topic_type)
        self._expected_sources = set(expected_sources)
        self._model_client = model_client
        # Tools are loaded so the LLM SEES them in its schema. They are NOT
        # invoked in this pass — `_render_proposed_actions` intercepts the
        # FunctionCalls and renders them instead.
        self._tools = {tool.name: tool for tool in tools}
        self._tool_schema = [tool.schema for tool in tools]
        # Single-threaded runtime + one event at a time => a bare dict is fine.
        # When events become concurrent this needs to be keyed by event_id.
        self._pending: Optional[Dict[str, str]] = None

    @message_handler
    async def handle_agent_response(
        self, message: AgentResponse, ctx: MessageContext
    ) -> None:
        # Identify the sender. `source_agent` is the preferred field; we fall
        # back to `reply_to_topic_type` because the base AIAgent doesn't
        # currently populate `source_agent`.
        source_agent = message.source_agent or message.reply_to_topic_type

        if source_agent not in self._expected_sources:
            console.section(
                f"Logic :: ignoring unexpected reply from {source_agent}",
                color=console.yellow,
            )
            return

        if self._pending is None:
            self._pending = {}

        reply_text = _extract_text(message.context)
        self._pending[source_agent] = reply_text

        console.section(f"Logic :: reply <- {source_agent}", color=console.yellow)
        console.body(reply_text)
        console.progress(
            "specialists in",
            len(self._pending),
            len(self._expected_sources),
        )

        if set(self._pending.keys()) < self._expected_sources:
            return  # still waiting for the rest

        await self._decide_and_propose(ctx)

    async def _decide_and_propose(self, ctx: MessageContext) -> None:
        assert self._pending is not None
        # Snapshot the findings and clear `_pending` up front so an exception
        # during the LLM call doesn't leave stale state behind.
        responses = self._pending
        self._pending = None

        findings_text = "\n\n".join(
            f"### {agent}\n{reply}" for agent, reply in responses.items()
        )

        console.section("Logic :: deciding...", color=console.cyan)

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=LOGIC_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        "Three specialists have replied. Their findings:\n\n"
                        f"{findings_text}\n\n"
                        "Decide on the appropriate next action."
                    ),
                    source="orchestrator",
                ),
            ],
            tools=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )

        proposed_text = _render_proposed_actions(llm_result.content)
        console.final_answer_box(
            "Logic :: decision (propose-only)", proposed_text
        )
        # No downstream consumer yet — the final_answer_box is the user-visible
        # output. When the interpreter / write-back path is wired up, that's
        # where this proposal will go next.


async def register_logic_agent(
    runtime: SingleThreadedAgentRuntime,
    agent_topic_type: str,
    expected_sources: Set[str],
    model_client: ChatCompletionClient,
) -> LogicAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=LOGIC_TOOLS)

    agent = await LogicAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: LogicAgent(
            logic_topic_type=agent_topic_type,
            expected_sources=expected_sources,
            model_client=model_client,
            tools=tools,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=agent.type,
        )
    )
    return agent
