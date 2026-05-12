from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

from src.agents.context_research_agent import register_context_research_agent
from src.agents.evidence_analyst import register_evidence_analyst
from src.agents.logic import register_logic_agent
from src.agents.orchestration import register_orchestration_agent
from src.agents.process_state_analyst import register_process_state_analyst
from src.primitives.contracts import AgentsTask, AgentstopicTypes, EventSources

load_dotenv()

from autogen_core import SingleThreadedAgentRuntime, TopicId
from autogen_core.models import ChatCompletionClient

from src.common.llm.azure import get_azure_lm


async def _register_default_agents(
    single_threaded_runtime: SingleThreadedAgentRuntime,
    azure_llm: ChatCompletionClient,
) -> None:
    await register_process_state_analyst(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Owns the live process picture: task statuses and dependency edges.",
        agent_topic_type=AgentstopicTypes.PROCESS_STATE_ANALYST.value,
        user_topic_type=AgentstopicTypes.LOGIC.value,
    )
    await register_evidence_analyst(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Reads the communication trail (emails, slack, prior decisions) attached to a task.",
        agent_topic_type=AgentstopicTypes.EVIDENCE_ANALYST.value,
        user_topic_type=AgentstopicTypes.LOGIC.value,
    )
    await register_context_research_agent(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Traverses the dependency graph for blast radius and historical precedents.",
        agent_topic_type=AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        user_topic_type=AgentstopicTypes.LOGIC.value,
    )
    await register_orchestration_agent(
        single_threaded_runtime,
        agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        participant_topic_types=[
            AgentstopicTypes.PROCESS_STATE_ANALYST.value,
            AgentstopicTypes.EVIDENCE_ANALYST.value,
            AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        ],
    )
    await register_logic_agent(
        single_threaded_runtime,
        agent_topic_type=AgentstopicTypes.LOGIC.value,
        expected_sources={
            AgentstopicTypes.PROCESS_STATE_ANALYST.value,
            AgentstopicTypes.EVIDENCE_ANALYST.value,
            AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        },
        model_client=azure_llm,
    )


async def run_pipeline_once(
    event_text: str,
    *,
    model_client: Optional[ChatCompletionClient] = None,
) -> None:
    """Start a fresh runtime, register default agents, publish one event, wait until idle.

    If ``model_client`` is omitted, this function creates an Azure client with
    ``get_azure_lm()`` and closes it before returning. If a client is passed in,
    the caller owns its lifecycle (useful when replaying many events).
    """
    own_client = model_client is None
    azure_llm = model_client or get_azure_lm()
    single_threaded_runtime = SingleThreadedAgentRuntime()
    try:
        await _register_default_agents(single_threaded_runtime, azure_llm)
        single_threaded_runtime.start()
        print("Agents registered successfully")

        await single_threaded_runtime.publish_message(
            AgentsTask(context=[event_text], source=EventSources.USER_CHAT),
            topic_id=TopicId(AgentstopicTypes.ORCHESTRATION.value, source="runtime"),
        )

        await single_threaded_runtime.stop_when_idle()
    finally:
        if own_client:
            await azure_llm.close()


async def main(input_method: callable) -> None:
    event_text = input_method("Please enter the event description: ")
    await run_pipeline_once(event_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(input_method=input))
