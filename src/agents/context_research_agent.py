"""ContextResearchAgent — explores the wider context graph (n-hop neighborhood,
related tasks, historical decisions) to surface non-obvious blast radius."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.agents.prompts import get_prompt_manager
from src.mcp.client import MCPClient

CONTEXT_RESEARCH_TOOLS = [
    "get_task_context",
    "validate_query",
    "find_similar_decisions",
]


async def register_context_research_agent(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=CONTEXT_RESEARCH_TOOLS)

    # Fetch the system_message from the YAML registry. Contracts are handled
    # separately via a Pydantic model and are not part of the system prompt.
    p = get_prompt_manager().get("context_research_agent")

    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: AIAgent(
            description=description,
            system_message=SystemMessage(content=p.system_message),
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
