import json
import sys
import base64
from pathlib import Path
from typing import Any

from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)

TIMEOUT = 60


class MCPToolWrapper:
    """Wraps an MCP tool to inject the session context into each tool call"""

    def __init__(self, inner_tool: Any):
        self._inner = inner_tool
        self.name = getattr(inner_tool, "name", "unknown")
        self.state_description = (
            getattr(inner_tool, "description", None) or "Processing request..."
        )

    @property
    def schema(self):
        """Returns the schema of the tool [name, description, state_description]"""
        return getattr(self._inner, "schema", {"type": "object", "properties": {}})

    async def run_json(self, arguments: dict, cancellation_token: Any = None) -> Any:
        """runs the tool with the given arguments and cancellation token"""
        return await self._inner.run_json(arguments, cancellation_token)

    def format_result(self, result: Any) -> str:
        """checks if the result is a string or a dictionary and formats it accordingly"""
        if result is None:
            return "No result returned."
        if isinstance(result, str):
            return result
        inner = getattr(self._inner, "return_value_as_string", None)
        return inner(result) if inner else json.dumps(result, default=str)

    def return_value_as_string(self, result: Any) -> str:
        """returns the result as a string"""
        return self.format_result(result)


class MCPClient:
    """Client for the local FastMCP server in ``server.py`` (stdio subprocess)."""

    def __init__(
        self,
        *,
        server_script: Path | str | None = None,
        server_params: StdioServerParams | None = None,
        timeout: float | int = TIMEOUT,
    ) -> None:
        if server_params is not None:
            self._server_params = server_params
        else:
            script = Path(__file__).resolve().parent / (
                server_script if server_script is not None else "pmo_server.py"
            )

            self._server_params = StdioServerParams(
                command=sys.executable,
                args=[str(script)],
                read_timeout_seconds=float(timeout),
            )

    def get_server_params(self) -> StdioServerParams:
        return self._server_params

    async def get_tools(
        self, exclude: list[str] | None = None, include: list[str] | None = None
    ) -> list[MCPToolWrapper]:
        """Returns a list of MCP tools wrapped with the session context"""

        server_params = self.get_server_params()
        if server_params is None:
            raise ValueError("Failed to get server params")

        inner_tools = await mcp_server_tools(server_params)
        wrapped = [MCPToolWrapper(tool) for tool in inner_tools]
        if include is not None:
            allowed = set(include)
            wrapped = [tool for tool in wrapped if tool.name in allowed]
        if exclude is not None:
            blocked_tool = set(exclude)
            wrapped = [tool for tool in wrapped if tool.name not in blocked_tool]
        return wrapped

    async def ping(self) -> None:
        """Checking the MCP server is reachable"""
        server_params = self.get_server_params()
        async with create_mcp_server_session(server_params) as session:
            await session.initialize()

    async def read_resource_text(self, uri: str) -> str:

        server_params = self.get_server_params()
        async with create_mcp_server_session(server_params) as session:
            await session.initialize()
            result = await session.read_resource(uri)

        parts: list[str] = []
        for block in result.contents:
            if hasattr(block, "text") and block.text is not None:
                parts.append(block.text)
            elif hasattr(block, "blob") and block.blob:
                parts.append(
                    base64.b64decode(block.blob).decode("utf-8", errors="replace")
                )
        return "\n".join(parts)

    async def run_tool(
        self,
        tool_name: str,
        arguments: dict | None = None,
        return_json: bool = False,
    ) -> Any:
        server_params = self.get_server_params()
        async with create_mcp_server_session(server_params) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text") and block.text is not None:
                parts.append(block.text)
        raw_output = "\n".join(parts)
        if not return_json:
            return raw_output
        if not raw_output.strip():
            return None
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
