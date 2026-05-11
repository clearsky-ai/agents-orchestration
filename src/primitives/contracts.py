from pydantic import BaseModel
from enum import Enum


class EventSources(Enum):
    USER_CHAT = "user_chat"
    AGENT = "agent"
    RECURRING_TASK = "recurring_task"


class AgentsTask(BaseModel):

    content: str
    source: EventSources


class AgentResponse(BaseModel):
    context: str
    reply_to_topic_type: str
