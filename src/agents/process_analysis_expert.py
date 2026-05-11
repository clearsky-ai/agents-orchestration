from typing import List
from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from base import AIAgent


async def register_process_analysis_expert(
    runtime: SingleThreadedAgentRuntime, 
    description: str,
    model_client: models.ChatCompletionClient,
    tools: List[models.Tool],
    delegate_tools: List[models.Tool],
    agent_topic_type: str,
    user_topic_type: str,
    output_channel_publish_method: callable,
    input_channel_subscribe_method: callable,
) -> AIAgent:

    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description=description,  # prompt_object.prompts.description,
            system_message="You are a process analysis expert agent that can analyze processes and provide insights.",
            model_client=model_client,
            tools=tools,
            delegate_tools=delegate_tools,
            agent_topic_type=agent_topic_type,
            user_topic_type=user_topic_type,
            output_channel_publish_method=output_channel_publish_method,
            input_channel_subscribe_method=input_channel_subscribe_method,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=agent.type,
        )
    )
    return agent