from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class EventSources(Enum):
    USER_CHAT = "user_chat"
    AGENT = "agent"
    RECURRING_TASK = "recurring_task"


class AgentsTask(BaseModel):
    # First entry is the raw event/user text; later entries are LLM transcript messages.
    context: List[Any]
    source: EventSources


class AgentResponse(BaseModel):
    context: Union[str, List[Any]]
    reply_to_topic_type: str
    # Which specialist produced this response (set by the base AIAgent).
    source_agent: Optional[str] = None


class AgentstopicTypes(Enum):
    ORCHESTRATION = "orchestration"
    PROCESS_STATE_ANALYST = "process_state_analyst"
    EVIDENCE_ANALYST = "evidence_analyst"
    CONTEXT_RESEARCH_AGENT = "context_research_agent"
    LOGIC = "logic"
    EXECUTION = "execution"
    # Terminal topic for the Execution's post-execution AgentResponse.
    # No agent subscribes — the message is dropped. Reserved as the
    # extension point for an audit / dashboard consumer later.
    EXECUTION_DONE = "execution.done"
    