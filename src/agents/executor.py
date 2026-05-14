from __future__ import annotations

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.agents.prompts import get_prompt_manager
from src.common import console
from src.mcp.client import MCPClient


def _notify(text: str) -> None:
    """Emit a user-facing notification after the executor finishes.

    Hooked into the executor via ``completion_callback`` on ``AIAgent``.
    Swap the body for a real channel (Slack / email / UI / webhook) when
    ready; the call-site doesn't change.
    """
    console.final_answer_box("Notification :: action taken", text)


EXECUTOR_TOOLS = [
    "update_task_status",
]


async def register_executor_agent(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=EXECUTOR_TOOLS)

    # Fetch system_message + contract from the YAML registry. Contract goes
    # LAST so it's the freshest thing the LLM saw before generating output.
    p = get_prompt_manager().get("executor")
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
            # Feed the executor's final LLM summary into the user-facing
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
