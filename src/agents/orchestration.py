from typing import List
import asyncio


from autogen_core import MessageContext, RoutedAgent, TopicId
from autogen_core import message_handler
from autogen_core.models import ChatCompletionClient

from src.primitives.contracts import AgentsTask


class OrchestrationAgent(RoutedAgent):

    def __init__(
        self,
        orchestration_topic_type: str,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        # participant_descriptions: List[str], # TODO: add this when we want to choose certain agents to participate in the orchestration.
    ) -> None:
        super().__init__(orchestration_topic_type)
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client

    @message_handler
    async def handle_task(self, task: AgentsTask, ctx: MessageContext) -> None:

        # broadcast to all agents.
        # Note that we use publish_message not send_message here.

        for participant in self._participant_topic_types:

            results = await asyncio.gather(
                *[
                    self.publish_message(
                        task, topic_id=TopicId(participant, source=self.id.key)
                    )
                    for participant in self._participant_topic_types
                ]
            )
        return results
