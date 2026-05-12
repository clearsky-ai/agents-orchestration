"""EvidenceAnalyst — reads the communication trail (emails, slack, prior decisions)
attached to a task and explains the narrative behind a signal."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.mcp.client import MCPClient

EVIDENCE_ANALYST_TOOLS = [
    "get_evidence_trace",
    "find_similar_decisions",
]

SYSTEM_PROMPT = """You are the EvidenceAnalyst.

Your job is to ground the email/signal in the **evidence trail** of the related task(s):
- Pull recent evidence rows (emails, slack notes, observations) with get_evidence_trace.
- Check for prior decisions on the same task with find_similar_decisions (precedents,
  manual overrides, variance tolerances).
- If the orchestrator's plan asks you to attach new evidence, use link_evidence.
- Cite evidence_id and decision_id values verbatim. Quote the relevant summary line
  so the orchestrator can fuse it with the process state and graph context.

Be terse: facts + citations, not commentary."""


async def register_evidence_analyst(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=EVIDENCE_ANALYST_TOOLS)

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
