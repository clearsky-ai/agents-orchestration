from dotenv import load_dotenv

load_dotenv()

import json
import os
import asyncio
from pathlib import Path
from copy import deepcopy
from autogen_core.models import UserMessage
from traceback import format_exc
from typing import Any, Optional, Callable
from autogen_core import SingleThreadedAgentRuntime, TopicId
from tenacity import retry, wait_exponential, stop_after_attempt

from src.mcp.client import MCPClient
from src.common.llm.dspy import get_lm
from src.constants import SYSTEM_SESSION_ID
from src.common.llm.azure import get_azure_lm
from src.primitives.session import SessionContext, DEFAULT_SESSION_CONTEXT
from src.tools.rag.glossary_retrieval import GlossaryRetrievalTool
from src.primitives.dtos import (
    ChatInputMessage,
    SessionStart,
    SessionStates,
    OutputChannelMessage,
)
from src.utils.logging.console_logging import (
    log_main_step,
    log_internal_step,
    log_internal_warning,
    log_error,
    ErrorsSeverity,
)
from src.utils.memory import (
    take_snapshot,
    log_session_memory_diff,
    log_session_memory_still_held,
)
from src.runtime_config import (
    agent2registry,
    active_agents as default_active_agents,
    crm_agent_topic,
    user_topic,
    primary_agent_topic,
    orchestrator_topic,
    broadcast_agent_topics,
)


def console_log(message: OutputChannelMessage):
    """Log a message to the console with formatting.

    Args:
        message: The output channel message to log.
    """
    print("-" * 20)
    print("-" * 20)
    print(str(message))
    print("-" * 20)
    print("-" * 20)



# we start with vanilla runtime



if __name__ == "__main__":
    
    pass

    
