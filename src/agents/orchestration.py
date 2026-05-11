"""Orchestration agent.

Flow:
1. Receives an `AgentsTask` from the dispatcher (email + interpreted signal).
2. Asks the LLM to produce a **per-agent plan** (one paragraph each) for the three
   specialists: ProcessStateAnalyst, EvidenceAnalyst, ContextResearchAgent.
3. Broadcasts the task in parallel, injecting each plan into its target topic.
4. Aggregates the three `AgentResponse` messages as they arrive.
5. Once all three are in, runs a final LLM call that synthesizes the findings and
   **proposes the next transition / actions**, then publishes the answer to the
   dispatcher.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)

from src.common import console
from src.primitives.contracts import AgentResponse, AgentsTask, EventSources


PLANNER_SYSTEM_PROMPT = """You are the ProcessOrchestrator's planner.

You are given an interpreted email/signal about a quarter-close process. Three
specialist agents will work in parallel on it:

- ProcessStateAnalyst: knows current task states, dependencies, and can update task
  attributes. Tools: process_status, run_sql_query, get_task_dependencies,
  explain_blocker, update_task_attribute, update_task_field.
- EvidenceAnalyst: reads emails/slack/decisions tied to a task. Tools:
  get_evidence_trace, link_evidence, find_similar_decisions.
- ContextResearchAgent: traverses the dependency graph for blast radius and
  historical precedents. Tools: get_task_context, validate_query,
  find_similar_decisions.

Produce a tight, actionable plan for EACH agent: which task_id(s) to focus on, which
tools to call (and with what arguments), and what to return. Do NOT answer the
question yourself — only plan.

Output STRICT JSON, no markdown fences, with exactly these keys:
{
  "process_state_analyst": "<plan>",
  "evidence_analyst": "<plan>",
  "context_research_agent": "<plan>"
}"""


SYNTHESIZER_SYSTEM_PROMPT = """You are the ProcessOrchestrator.

You sent the same email/signal to three specialists. Each replied with their findings.
Fuse their answers into a single response for the user. You MUST:

1. Briefly restate what the signal/email means in one sentence.
2. Summarize each specialist's key finding (1-2 lines each), citing task_ids,
   evidence_ids, decision_ids where relevant.
3. Propose the next transition / actions explicitly: list 1-5 concrete next steps,
   each labeled with the owner team (e.g. Payroll, Accounting Operations) and the
   task_id it affects. If a state transition is appropriate (e.g. T05 -> complete,
   T06 unblock), say so.
4. Flag open risks or missing information.

Be decisive, structured, and short. No emojis."""


def _extract_text(items: Any) -> str:
    """Pull the user-visible text out of an AgentResponse.context payload."""
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


def _extract_user_signal(context: List[Any]) -> str:
    """The dispatcher prepends the raw user/email string; recover it for planning."""
    for item in context:
        if isinstance(item, str):
            return item
        content = getattr(item, "content", None)
        if isinstance(content, str):
            return content
    return ""


def _parse_plan_json(raw: str) -> Dict[str, str]:
    """Strip code fences if the LLM added them, then parse JSON."""
    s = raw.strip()
    s = re.sub(r"^\s*```(?:json)?\s*\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return {}
    return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}


class OrchestrationAgent(RoutedAgent):
    def __init__(
        self,
        orchestration_topic_type: str,
        dispatcher_topic_type: str,
        participants: Dict[str, str],
        model_client: ChatCompletionClient,
    ) -> None:
        """
        participants: maps the planner JSON key -> the participant's topic type.
                      e.g. {"process_state_analyst": "process_state_analyst", ...}
        """
        super().__init__(orchestration_topic_type)
        self._orchestration_topic_type = orchestration_topic_type
        self._dispatcher_topic_type = dispatcher_topic_type
        self._participants = participants
        self._model_client = model_client
        # State for the currently-active task. Single-threaded runtime => one at a time.
        self._pending: Optional[Dict[str, Any]] = None

    async def _plan_for_participants(
        self, signal_text: str, cancellation_token
    ) -> Dict[str, str]:
        """Ask the LLM for one plan per specialist; fall back to a generic plan."""
        result = await self._model_client.create(
            messages=[
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        "Interpreted signal / email:\n"
                        f"{signal_text}\n\n"
                        "Return the JSON now."
                    ),
                    source="dispatcher",
                ),
            ],
            cancellation_token=cancellation_token,
        )
        plans = _parse_plan_json(result.content if isinstance(result.content, str) else "")
        fallback = (
            "Read the signal above, identify the task_id(s) it concerns, run the "
            "tools available to you, and return concise findings."
        )
        return {key: plans.get(key, fallback) for key in self._participants}

    @message_handler
    async def handle_task(
        self, message: AgentsTask, ctx: MessageContext
    ) -> None:
        signal_text = _extract_user_signal(message.context)
        console.banner("Orchestrator :: received signal")
        console.body(signal_text)

        plans = await self._plan_for_participants(signal_text, ctx.cancellation_token)
        console.banner("Orchestrator :: per-agent plans")
        console.render_plans(plans)

        self._pending = {
            "signal": signal_text,
            "expected": set(self._participants.values()),
            "responses": {},
            "plans": plans,
        }

        # Fan out the task to every specialist in parallel, each with its tailored plan.
        await asyncio.gather(
            *[
                self.publish_message(
                    AgentsTask(
                        context=list(message.context),
                        source=message.source,
                        plan=plans.get(plan_key),
                        target_agent=topic_type,
                    ),
                    topic_id=TopicId(topic_type, source=self.id.key),
                )
                for plan_key, topic_type in self._participants.items()
            ]
        )

    @message_handler
    async def handle_agent_response(
        self, message: AgentResponse, ctx: MessageContext
    ) -> None:
        if self._pending is None:
            print(
                "Orchestrator received an unexpected agent response (no pending task).",
                flush=True,
            )
            return

        source_agent = message.source_agent or message.reply_to_topic_type
        reply_text = _extract_text(message.context)
        self._pending["responses"][source_agent] = reply_text
        console.section(f"reply <- {source_agent}", color=console.yellow)
        console.body(reply_text)
        console.progress(
            "specialists in",
            len(self._pending["responses"]),
            len(self._pending["expected"]),
        )

        if set(self._pending["responses"].keys()) < self._pending["expected"]:
            return  # still waiting for the others

        await self._synthesize_and_reply(ctx)

    async def _synthesize_and_reply(self, ctx: MessageContext) -> None:
        assert self._pending is not None
        signal = self._pending["signal"]
        responses = self._pending["responses"]
        plans = self._pending["plans"]

        specialist_dump = "\n\n".join(
            f"### {agent}\nPlan it was given: {plans.get(_topic_to_plan_key(agent, self._participants), '(n/a)')}\n"
            f"Reply:\n{reply}"
            for agent, reply in responses.items()
        )

        synth = await self._model_client.create(
            messages=[
                SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                UserMessage(
                    content=(
                        f"Original interpreted signal / email:\n{signal}\n\n"
                        f"Specialist findings:\n{specialist_dump}\n\n"
                        "Now write the final answer and propose the next "
                        "transition/actions."
                    ),
                    source="orchestrator",
                ),
            ],
            cancellation_token=ctx.cancellation_token,
        )

        final_text = synth.content if isinstance(synth.content, str) else str(synth.content)
        console.final_answer_box("Orchestrator :: final answer", final_text)

        # Reset state BEFORE publishing so the dispatcher's next-input prompt does
        # not race a stale _pending.
        self._pending = None

        await self.publish_message(
            AgentResponse(
                context=[AssistantMessage(content=final_text, source=self.id.type)],
                reply_to_topic_type=self._orchestration_topic_type,
                source_agent=self.id.type,
            ),
            topic_id=TopicId(self._dispatcher_topic_type, source=self.id.key),
        )


def _topic_to_plan_key(topic_type: str, participants: Dict[str, str]) -> str:
    for plan_key, t in participants.items():
        if t == topic_type:
            return plan_key
    return topic_type


async def register_orchestration_agent(
    runtime: SingleThreadedAgentRuntime,
    model_client: ChatCompletionClient,
    agent_topic_type: str,
    dispatcher_topic_type: str,
    participants: Dict[str, str],
) -> OrchestrationAgent:
    orchestration_agent = await OrchestrationAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: OrchestrationAgent(
            orchestration_topic_type=agent_topic_type,
            dispatcher_topic_type=dispatcher_topic_type,
            participants=participants,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=orchestration_agent.type,
        )
    )
    return orchestration_agent
