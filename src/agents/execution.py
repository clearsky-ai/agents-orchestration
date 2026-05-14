from __future__ import annotations

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.common import console
from src.mcp.client import MCPClient


def _notify(text: str) -> None:
    """Emit a user-facing notification after the execution finishes.

    Hooked into the execution via ``completion_callback`` on ``AIAgent``.
    Swap the body for a real channel (Slack / email / UI / webhook) when
    ready; the call-site doesn't change.
    """
    console.final_answer_box("Notification :: action taken", text)


EXECUTION_TOOLS = [
    "update_task_status",
]


SYSTEM_PROMPT = """You are the ExecutionAgent.

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


async def register_execution_agent(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=EXECUTION_TOOLS)

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
            # Feed the execution's final LLM summary into the user-facing
            # notification (defined locally above). Other agents don't pass
            # a callback, so their replies don't pollute the notification.
            completion_callback=_notify,
        ),
    )

    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=agent.type,
        )
    )
    return agent
