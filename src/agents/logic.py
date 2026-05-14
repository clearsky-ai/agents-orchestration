from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)

from src.agents.policy_criticiser import critique
from src.common import console
from src.primitives.contracts import AgentResponse, AgentsTask, EventSources

LOGIC_SYSTEM_PROMPT = """You are the LogicAgent.

You receive findings from three specialist analysts about a single event:
- ProcessStateAnalyst — task statuses and dependency edges.
- EvidenceAnalyst — communication trail, prior decisions, and reasons a task is blocked.
- ContextResearchAgent — broader graph context and historical precedents.

These findings are complementary: each analyst contributes a distinct slice of
the picture, and you should combine them rather than treat them as competing or
redundant sources.

Your job is to plan what action (if any) the system should take in response
to the event. You are a planner — describe what needs to happen. A downstream
ExecutionAgent will perform the tasks you assign to it.

SCHEMA (use these names and values verbatim in your plan):
- Task fields (writable): name, team, status, business_day, owner, description.
  Task status values: ready | in_progress | complete | blocked | not_ready.

For each task you propose, be explicit about the details:
- the task_id it applies to (verbatim from the findings)
- the field/attribute name
- the new value
- Any constraints or conditions that must be met

The Execution does NOT re-check preconditions, so every step you propose must
be grounded in the findings right now. Do not assume future events, and do
not treat approval/drafting of an action as proof it has happened. 

If no action is warranted, begin your response with the exact phrase
"No action proposed." followed by a brief explanation."""


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
    """The LogicAgent has no tools, so the LLM response is plain text.

    Defensive: if for some reason the response isn't a string, fall back
    to repr so we don't crash the pipeline on a malformed shape.
    """
    if isinstance(content, str):
        return content
    return f"(unrecognized response shape: {content!r})"


def _escalate_to_human(*, plan: str, failed: List[tuple[str, str]]) -> None:
    """Surface a blocked plan to a human reviewer.

    Today this is just a console box; later it becomes a real channel
    (ticket queue, Slack, email, dashboard entry). The shape of the body —
    plan plus the list of failed policies and reasons — is what a reviewer
    needs to either approve, amend, or reject.
    """
    failed_lines = "\n".join(f"  - {p}: {reason}" for p, reason in failed)
    body = (
        f"A plan was proposed but failed one or more policy checks. "
        f"Human review required.\n\n"
        f"Plan:\n{plan}\n\n"
        f"Failed policies:\n{failed_lines}"
    )
    console.final_answer_box("HumanEscalation :: review required", body)


class LogicAgent(RoutedAgent):
    def __init__(
        self,
        logic_topic_type: str,
        execution_topic_type: str,
        expected_sources: Set[str],
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(logic_topic_type)
        self._execution_topic_type = execution_topic_type
        self._expected_sources = set(expected_sources)
        self._model_client = model_client
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
            cancellation_token=ctx.cancellation_token,
        )

        proposed_text = _render_proposed_actions(llm_result.content)
        console.section("Logic :: action plan", color=console.cyan)
        console.body(proposed_text)

        # No-action case: nothing to critique, no Yes/No routing to make.
        # Treated as a successful pass through the pipeline — notify the user
        # that the system looked at the event and decided no change was needed.
        if proposed_text.startswith("No action proposed"):
            console.final_answer_box("Notification :: no action needed", proposed_text)
            return

        # Single-shot policy critique. The reasoning trace is discarded;
        # we only act on the per-policy verdicts.
        _reasoning, results = await critique(findings_text, proposed_text)

        console.section("Logic :: policy critique", color=console.cyan)
        for r in results:
            mark = "passed" if r.passed else "failed"
            console.kv(f"  {mark} {r.policy}", r.reason)

        if all(r.passed for r in results):
            decision_summary = "Yes — plan routed to ExecutionAgent."
            console.final_answer_box(
                "Logic :: decision",
                f"{proposed_text}\n\n{decision_summary}",
            )
            # Publish AFTER rendering the box so the execution's output appears
            # below the box rather than interleaved with it. The plan flows
            # as plain text in the AgentsTask context; the Execution's LLM
            # reads it and decides which write tools to invoke.
            await self.publish_message(
                AgentsTask(
                    context=[proposed_text],
                    source=EventSources.AGENT,
                ),
                topic_id=TopicId(self._execution_topic_type, source=self.id.key),
            )
        else:
            # No path: plan exists but at least one policy failed. Escalate
            # to a human reviewer with the plan and the failed verdicts.
            failed = [(r.policy, r.reason) for r in results if not r.passed]
            _escalate_to_human(plan=proposed_text, failed=failed)


async def register_logic_agent(
    runtime: SingleThreadedAgentRuntime,
    agent_topic_type: str,
    execution_topic_type: str,
    expected_sources: Set[str],
    model_client: ChatCompletionClient,
) -> LogicAgent:
    agent = await LogicAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: LogicAgent(
            logic_topic_type=agent_topic_type,
            execution_topic_type=execution_topic_type,
            expected_sources=expected_sources,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=agent.type,
        )
    )
    return agent
