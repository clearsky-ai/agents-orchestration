import json
from typing import Callable, List, Optional

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, FunctionTool
from src.common import console
from src.primitives.contracts import AgentResponse, AgentsTask
from src.mcp.client import MCPToolWrapper


def _task_context_to_llm_messages(context: List[object]) -> List[object]:
    """Turn user string turns into UserMessage; leave transcript messages as-is."""
    out: List[object] = []
    for item in context:
        if isinstance(item, str):
            out.append(UserMessage(content=item, source="user"))
        else:
            out.append(item)
    return out


class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[MCPToolWrapper],
        agent_topic_type: str,
        user_topic_type: str,
        completion_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type
        # Optional hook called with the agent's final reply text after the
        # AgentResponse has been published. Used by the execution to feed its
        # summary into the user-facing notification without coupling base.py
        # to notification state.
        self._completion_callback = completion_callback

    def _build_messages(self, message: AgentsTask) -> List[object]:
        """Prepend the system prompt to the task context."""
        return [self._system_message] + _task_context_to_llm_messages(message.context)

    @message_handler
    async def handle_task(self, message: AgentsTask, ctx: MessageContext) -> None:

        console.section(f"{self.id.type} :: task received", color=console.magenta)

        # Send the task to the LLM.
        llm_result = await self._model_client.create(
            messages=self._build_messages(message),
            tools=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )

        if isinstance(llm_result.content, str):
            console.kv("llm-initial", llm_result.content)
        else:
            tool_names = (
                ", ".join(getattr(c, "name", "?") for c in llm_result.content)
                if isinstance(llm_result.content, list)
                else str(llm_result.content)
            )
            console.kv("llm-initial (tool calls)", tool_names)
        # Process the LLM result.
        while isinstance(llm_result.content, list) and all(
            isinstance(m, FunctionCall) for m in llm_result.content
        ):
            tool_call_results: List[FunctionExecutionResult] = []
            # Process each function call.
            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools:
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(
                        arguments, ctx.cancellation_token
                    )
                    result_as_str = self._tools[call.name].return_value_as_string(
                        result
                    )
                    tool_call_results.append(
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result_as_str,
                            is_error=False,
                            name=call.name,
                        )
                    )

                else:
                    raise ValueError(f"Unknown tool: {call.name}")

            if len(tool_call_results) > 0:
                tool_summary = ", ".join(r.name for r in tool_call_results)
                console.kv(f"{self.id.type} tool results", tool_summary)
                # Make another LLM call with the results.
                message.context.extend(
                    [
                        AssistantMessage(
                            content=llm_result.content, source=self.id.type
                        ),
                        FunctionExecutionResultMessage(content=tool_call_results),
                    ]
                )
                llm_result = await self._model_client.create(
                    messages=self._build_messages(message),
                    tools=self._tool_schema,
                    cancellation_token=ctx.cancellation_token,
                )
                if isinstance(llm_result.content, str):
                    console.kv(f"{self.id.type} llm-next", llm_result.content)
                else:
                    next_tools = (
                        ", ".join(getattr(c, "name", "?") for c in llm_result.content)
                        if isinstance(llm_result.content, list)
                        else str(llm_result.content)
                    )
                    console.kv(f"{self.id.type} llm-next (tool calls)", next_tools)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(llm_result.content, str)
        console.section(f"{self.id.type} :: final reply", color=console.green)
        console.body(llm_result.content)
        # Publish ONLY the final answer, not the whole accumulated transcript.
        # The transcript still grows inside the tool-loop above (because the
        # LLM needs to see prior tool results on each next call), but downstream
        # consumers only care about the final reply.
        await self.publish_message(
            AgentResponse(
                context=llm_result.content,
                reply_to_topic_type=self._agent_topic_type,
                source_agent=self.id.type,
            ),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
        if self._completion_callback is not None:
            try:
                self._completion_callback(llm_result.content)
            except Exception:
                # Notification errors must never break the pipeline.
                pass
