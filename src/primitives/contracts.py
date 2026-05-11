from typing import List
from pydantic import BaseModel
from enum import Enum


class EventSources(Enum):
    USER_CHAT = "user_chat"
    AGENT = "agent"
    RECURRING_TASK = "recurring_task"


class AgentsTask(BaseModel):
    context: List[str]
    source: EventSources


class AgentResponse(BaseModel):
    context: str
    reply_to_topic_type: str


class ChatInput(BaseModel):
    content: str

    @property
    def source(self):
        return EventSources.USER_CHAT

    def __init__(self, **data):
        data.pop("source", None)
        super().__init__(**data)


class AgentstopicTypes(Enum):
    DISPATCHER = "dispatcher"
    ORCHESTRATION = "orchestration"
    PROCESS_ANALYSIS_EXPERT = "process_analysis_expert"
    USER = "user"
