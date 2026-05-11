from typing import List
from pydantic import BaseModel, Field

# from src.primitives.contracts import AgentsTopicTypes


class AgentMission(BaseModel):

    agent_id: str
    task: str = Field(description="The task to be executed by the agent")


class ExecutionPlan(BaseModel):

    reasoning: str = Field(description="The reasoning for the execution plan")
    plan: List[AgentMission] = Field(description="The execution plan")
