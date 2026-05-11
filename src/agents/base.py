import json
from typing import List

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
)
from autogen_core.tools import Tool, FunctionTool
from src.primitives.contracts import AgentResponse, AgentsTask
from src.mcp.client import MCPToolWrapper


class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[MCPToolWrapper],
        agent_topic_type: str,
        user_topic_type: str,
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: AgentsTask, ctx: MessageContext) -> None:
        # Send the task to the LLM.
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )

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
                print(f"{'-'*80}\n{self.id.type}:\n{tool_call_results}", flush=True)
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
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    cancellation_token=ctx.cancellation_token,
                )
                print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(llm_result.content, str)
        message.context.append(
            AssistantMessage(content=llm_result.content, source=self.id.type)
        )
        await self.publish_message(
            AgentResponse(
                context=message.context, reply_to_topic_type=self._agent_topic_type
            ),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
