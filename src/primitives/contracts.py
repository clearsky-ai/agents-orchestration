from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class EventSources(Enum):
    USER_CHAT = "user_chat"
    AGENT = "agent"
    RECURRING_TASK = "recurring_task"


class AgentsTask(BaseModel):
    # str user lines from the dispatcher; later entries are LLM transcript messages.
    context: List[Any]
    source: EventSources
    # Optional per-agent plan injected by the orchestrator before broadcast.
    # When set, the receiving agent uses it as additional task-specific guidance.
    plan: Optional[str] = None
    # Identifies which specialist the task is targeted at (for plan routing/logging).
    target_agent: Optional[str] = None


class AgentResponse(BaseModel):
    context: Union[str, List[Any]]
    reply_to_topic_type: str
    # Which specialist produced this response (set by the base AIAgent).
    source_agent: Optional[str] = None


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
    PROCESS_STATE_ANALYST = "process_state_analyst"
    EVIDENCE_ANALYST = "evidence_analyst"
    CONTEXT_RESEARCH_AGENT = "context_research_agent"
