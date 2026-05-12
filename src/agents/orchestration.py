"""Orchestration agent.

Thin fan-out. Receives an `AgentsTask` (an event) and publishes a copy of it to
every participant topic in parallel. That's it.

The orchestrator no longer plans, aggregates, or synthesizes. The specialist
agents reply directly to the LogicAgent (configured via their own
`user_topic_type`), which is where decisions and actions live.
"""

from __future__ import annotations

import asyncio
from typing import List

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)

from src.common import console
from src.primitives.contracts import AgentsTask


ORCHESTRATOR_PREAMBLE = (
    "A new event has been routed to you. You are one of several specialist "
    "agents investigating it in parallel.\n\n"
    "For this event:\n"
    "- Use your tools to gather information relevant to your specialty.\n"
    "- Be concise; cite any identifiers verbatim where they appear in your "
    "findings.\n"
    "- Return only findings — do not propose actions or decisions. A "
    "downstream agent will handle that once all specialists have replied.\n\n"
    "Event:\n"
)


class OrchestrationAgent(RoutedAgent):
    def __init__(
        self,
        orchestration_topic_type: str,
        participant_topic_types: List[str],
    ) -> None:
        super().__init__(orchestration_topic_type)
        self._orchestration_topic_type = orchestration_topic_type
        self._participant_topic_types = participant_topic_types

    @message_handler
    async def handle_task(self, message: AgentsTask, ctx: MessageContext) -> None:
        console.banner("Orchestrator :: fan-out")
        console.body(f"-> {', '.join(self._participant_topic_types)}")

        # Wrap the raw event payload with a constant framing message so every
        # specialist receives identical instructions about what this task is.
        event_text = str(message.context[0]) if message.context else ""
        framed = ORCHESTRATOR_PREAMBLE + event_text

        await asyncio.gather(
            *[
                self.publish_message(
                    AgentsTask(
                        context=[framed],
                        source=message.source,
                    ),
                    topic_id=TopicId(topic_type, source=self.id.key),
                )
                for topic_type in self._participant_topic_types
            ]
        )


async def register_orchestration_agent(
    runtime: SingleThreadedAgentRuntime,
    agent_topic_type: str,
    participant_topic_types: List[str],
) -> OrchestrationAgent:
    orchestration_agent = await OrchestrationAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: OrchestrationAgent(
            orchestration_topic_type=agent_topic_type,
            participant_topic_types=participant_topic_types,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=orchestration_agent.type,
        )
    )
    return orchestration_agent
