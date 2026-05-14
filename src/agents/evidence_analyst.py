"""EvidenceAnalyst — reads the communication trail (emails, slack, prior decisions)
attached to a task and explains the narrative behind a signal."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.agents.prompts import get_prompt_manager
from src.mcp.client import MCPClient

EVIDENCE_ANALYST_TOOLS = [
    "get_evidence_trace",
    "find_similar_decisions",
    "explain_blocker",
]


async def register_evidence_analyst(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=EVIDENCE_ANALYST_TOOLS)

    # Fetch system_message + contract from the YAML registry. Contract goes
    # LAST so it's the freshest thing the LLM saw before generating output.
    p = get_prompt_manager().get("evidence_analyst")
    prompt = (
        f"{p.system_message}\n\n"
        f"# Contract — self-check your output before returning\n"
        f"{p.contract}"
    )

    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: AIAgent(
            description=description,
            system_message=SystemMessage(content=prompt),
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
