from typing import List
import asyncio


from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core import message_handler
from autogen_core.models import ChatCompletionClient

from src.primitives.contracts import AgentsTask


class OrchestrationAgent(RoutedAgent):

    def __init__(
        self,
        orchestration_topic_type: str,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        # participant_descriptions: List[str],
        # TODO: add when choosing which agents participate in orchestration.
    ) -> None:
        super().__init__(orchestration_topic_type)
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client

    @message_handler
    async def handle_task(
        self, message: AgentsTask, ctx: MessageContext
    ) -> None:

        # broadcast to all agents.
        # Note that we use publish_message not send_message here.

        await asyncio.gather(
            *[
                self.publish_message(
                    message,
                    topic_id=TopicId(
                        participant_topic, source=self.id.key
                    ),
                )
                for participant_topic in self._participant_topic_types
            ]
        )


async def register_orchestration_agent(
    runtime: SingleThreadedAgentRuntime,
    model_client: ChatCompletionClient,
    agent_topic_type: str,
    participant_topic_types: List[str],
) -> OrchestrationAgent:
    orchestration_agent = await OrchestrationAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: OrchestrationAgent(
            orchestration_topic_type=agent_topic_type,
            participant_topic_types=participant_topic_types,
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
