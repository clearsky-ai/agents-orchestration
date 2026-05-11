from autogen_core import MessageContext, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core import RoutedAgent, TopicId, message_handler
from autogen_core.models import UserMessage

from src.agents.base import AIAgent
from src.primitives.contracts import AgentResponse, AgentsTask, ChatInput


class DispatcherAgent(RoutedAgent):
    def __init__(
        self,
        dispatcher_topic_type: str,
        orchestration_agent_topic_type: str,
        input_channel_subscribe_method: callable,
        output_channel_publish_method: callable,
    ):
        super().__init__(dispatcher_topic_type)
        self._orchestration_agent_topic_type = orchestration_agent_topic_type
        self._input_channel_subscribe_method = input_channel_subscribe_method
        self._output_channel_publish_method = output_channel_publish_method

    @message_handler
    async def handle_chat_input(self, message: ChatInput, ctx: MessageContext) -> None:

        print(f"Publishing task to orchestration agent: {message.content}")
        # publish to orchestration agent
        await self.publish_message(
            AgentsTask(context=[message.content], source=message.source),
            topic_id=TopicId(self._orchestration_agent_topic_type, source=self.id.key),
        )

    @message_handler
    async def handle_agent_response(
        self, message: AgentResponse, topic_id: TopicId
    ) -> None:

        self._output_channel_publish_method(message.context)

        next_input = self._input_channel_subscribe_method()
        if next_input.strip() == "" or next_input.strip().lower() == "Fuck off":

            self._output_channel_publish_method(
                AgentResponse(
                    context="Goodbye",
                    reply_to_topic_type=self._orchestration_agent_topic_type,
                )
            )
            return
        await self.publish_message(
            ChatInput(content=next_input),
            topic_id=TopicId(self._orchestration_agent_topic_type, source=self.id.key),
        )


async def register_dispatcher_agent(
    runtime: SingleThreadedAgentRuntime,
    agent_topic_type: str,
    orchestration_agent_topic_type: str,
    input_channel_subscribe_method: callable,
    output_channel_publish_method: callable,
) -> DispatcherAgent:

    dispatcher_agent = await DispatcherAgent.register(
        runtime,
        type=agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: DispatcherAgent(
            dispatcher_topic_type=agent_topic_type,
            orchestration_agent_topic_type=orchestration_agent_topic_type,
            input_channel_subscribe_method=input_channel_subscribe_method,
            output_channel_publish_method=output_channel_publish_method,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=dispatcher_agent.type,
        )
    )
    return dispatcher_agent
