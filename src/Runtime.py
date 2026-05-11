from dotenv import load_dotenv

from src.agents.dispatcher import register_dispatcher_agent
from src.agents.orchestration import register_orchestration_agent
from src.agents.process_analysis_expert import register_process_analysis_expert
from src.agents.context_graph_expert import register_context_graph_expert
from src.primitives.contracts import AgentsTopicTypes, ChatInput

load_dotenv()

from src.common.llm.azure import get_azure_lm
from src.common.llm.dspy import get_lm
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
        agent_topic_type=AgentsTopicTypes.DISPATCHER.value,
        orchestration_agent_topic_type=AgentsTopicTypes.ORCHESTRATION.value,
        input_channel_subscribe_method=input_method,
        output_channel_publish_method=console_log,
    )

    # register process analysis expert agent:
    process_analysis_expert_agent = await register_process_analysis_expert(
        single_threaded_runtime,
        model_client=azure_llm,
        description="The process analysis expert agent is responsible for analyzing the processes and providing insights.",
        agent_topic_type=AgentsTopicTypes.PROCESS_ANALYSIS_EXPERT.value,
        user_topic_type=AgentsTopicTypes.DISPATCHER.value,
    )
    cg_expert_agent = await register_context_graph_expert(
        single_threaded_runtime,
        model_client=azure_llm,
        description="The context graph expert agent is responsible for analyzing the context graph and providing insights.",
        agent_topic_type=AgentsTopicTypes.CONTEXT_GRAPH_EXPERT.value,
        user_topic_type=AgentsTopicTypes.DISPATCHER.value,
    )
    # regisrter orchestration agent:
    orchestration_agent = await register_orchestration_agent(
        single_threaded_runtime,
        dspy_agent=get_lm(),
        agent_topic_type=AgentsTopicTypes.ORCHESTRATION.value,
        participant_topic_types=[AgentsTopicTypes.PROCESS_ANALYSIS_EXPERT.value],
    )
    single_threaded_runtime.start()
    print("Agents registered successfully")
    await single_threaded_runtime.publish_message(
        ChatInput(content="Hello!"),
        topic_id=TopicId(AgentsTopicTypes.DISPATCHER.value, source=dispatcher_agent),
    )

    await single_threaded_runtime.stop_when_idle()
    await azure_llm.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(input_method=input))
