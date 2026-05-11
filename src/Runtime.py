from dotenv import load_dotenv

from src.agents.dispatcher import register_dispatcher_agent
from src.agents.orchestration import register_orchestration_agent
from src.agents.process_analysis_expert import register_process_analysis_expert
from src.primitives.contracts import AgentstopicTypes, ChatInput

load_dotenv()

from src.common.llm.azure import get_azure_lm
from autogen_core import SingleThreadedAgentRuntime, TopicId


def console_log(message: str):
    """Log a message to the console with formatting.

    Args:
        message: The output channel message to log.
    """
    print("-" * 20)
    print("-" * 20)
    print(str(message))
    print("-" * 20)
    print("-" * 20)


async def main(input_method: callable):
    single_threaded_runtime = SingleThreadedAgentRuntime()

    azure_llm = get_azure_lm()
    # register dispatcher agent:
    dispatcher_agent = await register_dispatcher_agent(
        single_threaded_runtime,
        agent_topic_type=AgentstopicTypes.DISPATCHER.value,
        orchestration_agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        input_channel_subscribe_method=input_method,
        output_channel_publish_method=console_log,
    )

    # register process analysis expert agent:
    process_analysis_expert_agent = await register_process_analysis_expert(
        single_threaded_runtime,
        model_client=azure_llm,
        description="The process analysis expert agent is responsible for analyzing the processes and providing insights.",
        agent_topic_type=AgentstopicTypes.PROCESS_ANALYSIS_EXPERT.value,
        user_topic_type=AgentstopicTypes.DISPATCHER.value,
    )

    # regisrter orchestration agent:
    orchestration_agent = await register_orchestration_agent(
        single_threaded_runtime,
        model_client=azure_llm,
        agent_topic_type=AgentstopicTypes.ORCHESTRATION.value,
        participant_topic_types=[AgentstopicTypes.PROCESS_ANALYSIS_EXPERT.value],
    )
    single_threaded_runtime.start()
    print("Agents registered successfully")
    await single_threaded_runtime.publish_message(
        ChatInput(content="Hello!"),
        topic_id=TopicId(AgentstopicTypes.DISPATCHER.value, source=dispatcher_agent),
    )

    await single_threaded_runtime.stop_when_idle()
    await azure_llm.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(input_method=input))
