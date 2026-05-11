from dotenv import load_dotenv

from src.agents.context_research_agent import register_context_research_agent
from src.agents.dispatcher import register_dispatcher_agent
from src.agents.evidence_analyst import register_evidence_analyst
from src.agents.orchestration import register_orchestration_agent
from src.agents.process_state_analyst import register_process_state_analyst
from src.primitives.contracts import AgentResponse, AgentstopicTypes, ChatInput

load_dotenv()

from src.common import console
from src.common.llm.azure import get_azure_lm
from autogen_core import SingleThreadedAgentRuntime, TopicId


def _extract_dispatcher_text(payload) -> str:
    """The dispatcher hands us either a raw string, an AgentResponse, or its context list."""
    if isinstance(payload, AgentResponse):
        payload = payload.context
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = []
        for item in payload:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n\n".join(parts)
    return str(payload)


def console_log(message):
    """Render the dispatcher's outbound message in a readable box."""
    console.final_answer_box(
        "Dispatcher :: message to user", _extract_dispatcher_text(message)
    )


async def main(input_method: callable):
    single_threaded_runtime = SingleThreadedAgentRuntime()

    azure_llm = get_azure_lm()

    dispatcher_agent = await register_dispatcher_agent(
        single_threaded_runtime,
        agent_topic_type=AgentstopicTypes.DISPATCHER.value,
        orchestration_agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        input_channel_subscribe_method=input_method,
        output_channel_publish_method=console_log,
    )

    # Specialists — each replies to the orchestrator topic, not the dispatcher.
    await register_process_state_analyst(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Owns the live process picture: task states, dependency edges, and write-backs.",
        agent_topic_type=AgentstopicTypes.PROCESS_STATE_ANALYST.value,
        user_topic_type=AgentstopicTypes.ORCHESTRATION.value,
    )
    await register_evidence_analyst(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Reads the communication trail (emails, slack, prior decisions) attached to a task.",
        agent_topic_type=AgentstopicTypes.EVIDENCE_ANALYST.value,
        user_topic_type=AgentstopicTypes.ORCHESTRATION.value,
    )
    await register_context_research_agent(
        single_threaded_runtime,
        model_client=azure_llm,
        description="Traverses the dependency graph for blast radius and historical precedents.",
        agent_topic_type=AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        user_topic_type=AgentstopicTypes.ORCHESTRATION.value,
    )

    # Orchestrator: plans -> broadcast -> aggregate -> synthesize -> reply to dispatcher.
    await register_orchestration_agent(
        single_threaded_runtime,
        model_client=azure_llm,
        agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        dispatcher_topic_type=AgentstopicTypes.DISPATCHER.value,
        participants={
            "process_state_analyst": AgentstopicTypes.PROCESS_STATE_ANALYST.value,
            "evidence_analyst": AgentstopicTypes.EVIDENCE_ANALYST.value,
            "context_research_agent": AgentstopicTypes.CONTEXT_RESEARCH_AGENT.value,
        },
    )

    single_threaded_runtime.start()
    print("Agents registered successfully")
    user_input = input_method("Please enter your input: ")
    await single_threaded_runtime.publish_message(
        ChatInput(content=user_input),
        topic_id=TopicId(AgentstopicTypes.DISPATCHER.value, source=dispatcher_agent),
    )

    await single_threaded_runtime.stop_when_idle()
    await azure_llm.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(input_method=input))
