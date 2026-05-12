from dotenv import load_dotenv

from src.agents.context_research_agent import register_context_research_agent
from src.agents.evidence_analyst import register_evidence_analyst
from src.agents.logic import register_logic_agent
from src.agents.orchestration import register_orchestration_agent
from src.agents.process_state_analyst import register_process_state_analyst
from src.primitives.contracts import AgentsTask, AgentstopicTypes, EventSources

load_dotenv()

from src.common.llm.azure import get_azure_lm
from autogen_core import SingleThreadedAgentRuntime, TopicId


async def main(input_method: callable):
    single_threaded_runtime = SingleThreadedAgentRuntime()

    azure_llm = get_azure_lm()

    # Specialists — each replies to the LogicAgent topic.
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

    # Orchestrator: thin fan-out only. No LLM, no aggregation.
    await register_orchestration_agent(
        single_threaded_runtime,
        agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        participant_topic_types=[
            AgentstopicTypes.PROCESS_STATE_ANALYST.value,
            AgentstopicTypes.EVIDENCE_ANALYST.value,
            AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        ],
    )

    # LogicAgent: aggregates the three analyst replies, runs an LLM with the
    # write tools' schemas, and renders the LLM's decision as proposed actions.
    # Propose-only — tool calls are NOT executed.
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

    single_threaded_runtime.start()
    print("Agents registered successfully")

    # Inject one event into the pipeline. The interpreter will replace this
    # stdin prompt once it lands.
    event_text = input_method("Please enter the event description: ")
    await single_threaded_runtime.publish_message(
        AgentsTask(context=[event_text], source=EventSources.USER_CHAT),
        topic_id=TopicId(AgentstopicTypes.ORCHESTRATION.value, source="runtime"),
    )

    await single_threaded_runtime.stop_when_idle()
    await azure_llm.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(input_method=input))
