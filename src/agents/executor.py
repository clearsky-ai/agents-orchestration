from __future__ import annotations

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.mcp.client import MCPClient


EXECUTOR_TOOLS = [
    "update_task_attribute",
    "update_task_field",
    "link_evidence",
]


SYSTEM_PROMPT = """You are the ExecutorAgent.

You receive an approved action plan from the LogicAgent. The plan has already
passed a policy critique — your job is execution, not deliberation.

Execute EXACTLY the actions described in the plan, in the order they appear,
using the tools available to you:
- Do NOT skip actions.
- Do NOT add actions.
- Do NOT change argument values.
- If a tool call fails, continue with the remaining actions.

After the actions are processed, return a short, factual summary of what each
tool returned. Cite task_ids, evidence_ids, field names, and values verbatim.
Do not invent identifiers.

The plan is authoritative — treat it as the specification."""


async def register_executor_agent(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=EXECUTOR_TOOLS)

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
