from pathlib import Path
from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage
from src.agents.base import AIAgent
from src.mcp.client import MCPClient as CgMcpClient


async def register_context_graph_expert(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    cg_mcp_client = CgMcpClient(
        server_script="cg_server.py"
    )

    tools = await cg_mcp_client.get_tools()
    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description=description,  # prompt_object.prompts.description,
            system_message=SystemMessage(
                content="You are a context graph expert agent that can analyze the context graph and provide insights."
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
