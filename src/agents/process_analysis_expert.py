from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage
from src.agents.base import AIAgent
from src.mcp.client import MCPClient as PmoMcpClient


async def register_process_analysis_expert(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    pmo_mcp_client = PmoMcpClient()
    
    tools = await pmo_mcp_client.get_tools()
    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description=description,  # prompt_object.prompts.description,
            system_message=SystemMessage(
                content="You are a process analysis expert agent that can analyze processes and provide insights."
            ),
            model_client=model_client,
            tools=tools,
            agent_topic_type=agent_topic_type,
            user_topic_type=user_topic_type,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=agent.type,
        )
    )
    return agent
