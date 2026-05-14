from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from src.primitives.contracts import AgentstopicTypes


class ActionType(Enum):
    PMO_ACTION = "pmo_action"
    SEND_NOTIFICATION = "send_notification"
    HUMAN_ESCALATION = "human_escalation"
    NO_ACTION = "no_action"


class Mission(BaseModel):

    agent: AgentstopicTypes = Field(
        default=AgentstopicTypes.EXECUTION
    )  # * Each mission is assigned to a specific agent. Now we only have execution agent, but in the future we will have more agents.
    action_type: ActionType
    mission_task: str = Field(description="The task to be executed by the agent")
    reasoning: str = Field(description="The reasoning for why this action is being proposed")


class ExecutionPlan(BaseModel):

    missions: List[Mission] = Field(
        description="The missions to be executed by the execution agent"
    )
