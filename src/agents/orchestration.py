import dspy
import asyncio
from typing import Any, List

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core import message_handler
from autogen_core.models import ChatCompletionClient

from src.primitives.contracts import AgentsTask
from src.primitives.mission import ExecutionPlan, AgentMission


class PrepareMission(dspy.Signature):
    """
    Given a list of sub-agent experts and a query, prepare a mission that responds to the query.
    The mission is an ordered list of sub-agent tasks that are to be executed in sequence.
    Make sure to be comprehensive and list the expected results from the mission at the reasoning.
    """

    sub_agent_experts: List[str] = dspy.InputField(
        desc="The list of sub-agent experts to be used in the mission and their descriptions as key-value pairs"
    )
    query: str = dspy.InputField(desc="The query to be responded to")
    mission: List[AgentMission] = dspy.OutputField(desc="The mission to be executed")


class OrchestrationAgent(RoutedAgent):

    def __init__(
        self,
        orchestration_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        dspy_agent: any,
    ) -> None:
        super().__init__(orchestration_topic_type)
        self._participant_topic_types = participant_topic_types
        self._participant_descriptions = participant_descriptions
        self._model_client = dspy_agent

        self._mission_planner = dspy.ChainOfThought(PrepareMission)

    @message_handler
    async def handle_task(self, message: AgentsTask, ctx: MessageContext) -> None:

        # first: prepare a mission
        with dspy.context(lm=self._model_client):
            mission_planned = self._mission_planner(
                sub_agent_experts=[
                    f"{k}: {v}"
                    for k, v in zip(
                        self._participant_topic_types, self._participant_descriptions
                    )
                ],
                query=message.context[
                    -1
                ],  # TODO: consider handlig the history of the conversation.
            )

        plan = ExecutionPlan(
            reasoning=mission_planned.reasoning,
            plan=mission_planned.mission,
        )
        with open("plan.json", "w") as f:
            import json

            json.dump(plan.model_dump(), f, indent=4)

        # broadcast to all agents.
        # Note that we use publish_message not send_message here.

        await asyncio.gather(
            *[
                self.publish_message(
                    message,
                    topic_id=TopicId(participant_topic, source=self.id.key),
                )
                for participant_topic in self._participant_topic_types
            ]
        )


async def register_orchestration_agent(
    runtime: SingleThreadedAgentRuntime,
    dspy_agent: any,
    agent_topic_type: str,
    participant_topic_types: List[str],
    participant_descriptions: List[str],
) -> OrchestrationAgent:
    orchestration_agent = await OrchestrationAgent.register(
        runtime,
        type=agent_topic_type,
        factory=lambda: OrchestrationAgent(
            orchestration_topic_type=agent_topic_type,
            participant_topic_types=participant_topic_types,
            participant_descriptions=participant_descriptions,
            dspy_agent=dspy_agent,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(
            topic_type=agent_topic_type,
            agent_type=orchestration_agent.type,
        )
    )
    return orchestration_agent
