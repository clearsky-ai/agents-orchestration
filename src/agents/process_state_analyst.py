"""ProcessStateAnalyst — owns the live process picture: task states, dependency edges,
and write-backs to task attributes. Receives a focused subset of MCP tools."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.mcp.client import MCPClient

PROCESS_STATE_ANALYST_TOOLS = [
    "process_status",
    "run_sql_query",
    "get_task_dependencies",
    "explain_blocker",
    "update_task_attribute",
    "update_task_field",
]

SYSTEM_PROMPT = """You are the ProcessStateAnalyst.

Your job is to read the **current state of the quarter-close process** and tell the
orchestrator what is happening right now around the email/signal you are given:
- Which task(s) does the signal touch?
- What is each task's state (complete / in_progress / blocked / not_ready / ready)?
- What are the upstream and downstream dependencies, and are they satisfied?
- If something is blocked, why (use explain_blocker)?
- If the orchestrator's plan asks you to update a task attribute, do it.

Use only the tools you have been given. Cite task IDs (T01, T02, ...) in your answer.
Return a concise, structured finding the orchestrator can fuse with the other agents."""


async def register_process_state_analyst(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=PROCESS_STATE_ANALYST_TOOLS)

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
