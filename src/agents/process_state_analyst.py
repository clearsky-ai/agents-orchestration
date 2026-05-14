"""ProcessStateAnalyst — owns the live process picture: task states and dependency
edges. Read-only; the LogicAgent owns all write-backs. Receives a focused subset
of MCP tools."""

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription, models
from autogen_core.models import SystemMessage

from src.agents.base import AIAgent
from src.agents.prompts import get_prompt_manager
from src.mcp.client import MCPClient

PROCESS_STATE_ANALYST_TOOLS = [
    "process_status",
    "run_sql_query",
    "get_task_dependencies",
]


async def register_process_state_analyst(
    runtime: SingleThreadedAgentRuntime,
    description: str,
    model_client: models.ChatCompletionClient,
    agent_topic_type: str,
    user_topic_type: str,
) -> AIAgent:
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools(include=PROCESS_STATE_ANALYST_TOOLS)

    # Pull the live SQL schema once at registration and bake it into the
    # system prompt so the LLM doesn't have to guess column names when
    # drafting `run_sql_query` calls.
    schema_text = await mcp_client.read_resource_text("schema://sql")

    # Fetch the system_message from the YAML registry. Contracts are handled
    # separately via a Pydantic model and are not part of the system prompt.
    p = get_prompt_manager().get("process_state_analyst")
    prompt = (
        f"{p.system_message}\n\n"
        f"SQL schema (use these exact column names in run_sql_query):\n"
        f"{schema_text}"
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
