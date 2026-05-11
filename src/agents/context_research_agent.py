"""ContextResearchAgent — explores the wider context graph (n-hop neighborhood,
related tasks, historical decisions) to surface non-obvious blast radius."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.mcp.client import MCPClient

CONTEXT_RESEARCH_TOOLS = [
    "get_task_context",
    "validate_query",
    "find_similar_decisions",
]

SYSTEM_PROMPT = """You are the ContextResearchAgent.

Your job is to **widen the lens**: given the email/signal and the central task(s) the
orchestrator points you to, traverse the dependency graph and surface the broader context.
- Call get_task_context with a sensible hops value (1-3) to retrieve the n-hop
  neighborhood (upstream prerequisites AND downstream dependents).
- Look for previously similar decisions (find_similar_decisions) that should bias
  the recommendation (precedents, tolerances, manual overrides).
- If the orchestrator's plan provides a custom Cypher snippet, validate it with
  validate_query before trusting it.
- Report: which neighboring tasks are at risk, which historical decisions are relevant,
  and what the **blast radius** of the signal looks like.

Stay focused on graph-level context; do not duplicate the EvidenceAnalyst's quotes
or the ProcessStateAnalyst's state breakdown."""


async def register_context_research_agent(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=CONTEXT_RESEARCH_TOOLS)

    agent = await AIAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: AIAgent(
            description=description,
            system_message=SystemMessage(content=SYSTEM_PROMPT),
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
