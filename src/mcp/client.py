"""Async MCP stdio client for this repo's FastMCP server (`src.mcp.server.server`)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from src.mcp.cp_resources import EVIDENCE_BUNDLE_SCHEMA_URI, TASK_CONTEXT_SCHEMA_URI

__all__ = [
    "EVIDENCE_BUNDLE_SCHEMA_URI",
    "TASK_CONTEXT_SCHEMA_URI",
    "call_tool_json",
    "decode_tool_result",
    "stdio_server_params_agents_orchestration",
    "mcp_server_session",
    "repo_root",
]


def repo_root() -> Path:
    """Repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def stdio_server_params_agents_orchestration(
    *,
    root: Path | None = None,
    python_executable: str | None = None,
) -> StdioServerParameters:
    """Spawn the Context Graph / Context Platform MCP server over stdio.

    Runs ``python -c "from src.mcp.server import server; server.run()"`` with
    ``cwd`` and ``PYTHONPATH`` set to the repo root so ``src.*`` imports resolve.
    """
    r = root or repo_root()
    exe = python_executable or sys.executable
    code = "from src.mcp.server import server; server.run()"
    env = {**os.environ, "PYTHONPATH": str(r)}
    return StdioServerParameters(command=exe, args=["-c", code], cwd=str(r), env=env)


def _structured_content(result: types.CallToolResult) -> Any:
    """SDK versions differ: Pydantic may expose ``structuredContent`` or ``structured_content``."""
    sc = getattr(result, "structured_content", None)
    if sc is None:
        sc = getattr(result, "structuredContent", None)
    return sc


def decode_tool_result(result: types.CallToolResult) -> Any:
    """Return structured JSON/dict from a tool call, else the first text blob, else the raw result."""
    sc = _structured_content(result)
    if isinstance(sc, dict):
        return sc
    if sc is not None:
        return sc
    for block in result.content:
        if isinstance(block, types.TextContent):
            text = block.text.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return block.text
    return None


@asynccontextmanager
async def mcp_server_session(
    params: StdioServerParameters | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect to the MCP server over stdio and yield an initialized ``ClientSession``."""
    sp = params or stdio_server_params_agents_orchestration()
    async with stdio_client(sp) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool_json(
    session: ClientSession, name: str, arguments: dict[str, Any] | None = None
) -> Any:
    """``call_tool`` plus :func:`decode_tool_result`."""
    raw = await session.call_tool(name, arguments=arguments or {})
    return decode_tool_result(raw)


def _truncate_cypher_in_result(result: Any) -> Any:
    """Shorten ``cypher`` strings in ``get_task_context`` responses for readable demo output."""
    if not isinstance(result, dict) or not result.get("ok"):
        return result
    cy = result.get("cypher")
    if not isinstance(cy, str) or len(cy) <= 400:
        return result
    out = dict(result)
    out["cypher"] = f"{cy[:400]}... ({len(cy)} chars total)"
    return out


def _print_tool_case(name: str, args: dict[str, Any], result: Any) -> None:
    if name == "get_task_context":
        result = _truncate_cypher_in_result(result)
    print("\n" + "=" * 72)
    print(f"TOOL: {name}")
    print(f"ARGS: {json.dumps(args, default=str)}")
    print("RESULT:")
    print(json.dumps(result, indent=2, default=str))


async def _demo() -> None:
    async with mcp_server_session() as session:
        listed = await session.list_tools()
        names = {t.name for t in listed.tools}
        expected = {
            "get_task_context",
            "validate_query",
            "explain_blocker",
            "find_similar_decisions",
            "get_evidence_trace",
            "update_task_field",
            "link_evidence",
        }
        missing = sorted(expected - names)
        if missing:
            raise RuntimeError(f"MCP server missing tools {missing}; have {sorted(names)!r}")

        print("tools:", sorted(names))
        res = await session.list_resources()
        print("resources:", [str(r.uri) for r in res.resources])

        _print_tool_case(
            "validate_query",
            {"query": "MATCH (n) RETURN n LIMIT 1"},
            await call_tool_json(
                session, "validate_query", {"query": "MATCH (n) RETURN n LIMIT 1"}
            ),
        )

        _print_tool_case(
            "explain_blocker",
            {"task_id": "T06"},
            await call_tool_json(session, "explain_blocker", {"task_id": "T06"}),
        )

        _print_tool_case(
            "find_similar_decisions",
            {"task_id": "T06", "lookback_days": 90, "limit": 5},
            await call_tool_json(
                session,
                "find_similar_decisions",
                {"task_id": "T06", "lookback_days": 90, "limit": 5},
            ),
        )

        _print_tool_case(
            "get_evidence_trace",
            {"task_id": "T05", "code_query": "", "page": 1, "page_size": 10},
            await call_tool_json(
                session,
                "get_evidence_trace",
                {"task_id": "T05", "code_query": "", "page": 1, "page_size": 10},
            ),
        )

        _print_tool_case(
            "get_task_context",
            {"task_id": "T01", "hops": 1},
            await call_tool_json(
                session, "get_task_context", {"task_id": "T01", "hops": 1}
            ),
        )

        demo_field = "_mcp_client_demo_field"
        _print_tool_case(
            "update_task_field",
            {"task_id": "T01", "field": demo_field, "value": "mcp-client-was-here"},
            await call_tool_json(
                session,
                "update_task_field",
                {"task_id": "T01", "field": demo_field, "value": "mcp-client-was-here"},
            ),
        )

        # ev-001 is linked to T05 in mock_data.json; move to T01 then restore.
        _print_tool_case(
            "link_evidence",
            {"task_id": "T01", "evidence_id": "ev-001"},
            await call_tool_json(
                session, "link_evidence", {"task_id": "T01", "evidence_id": "ev-001"}
            ),
        )
        _print_tool_case(
            "link_evidence (restore)",
            {"task_id": "T05", "evidence_id": "ev-001"},
            await call_tool_json(
                session, "link_evidence", {"task_id": "T05", "evidence_id": "ev-001"}
            ),
        )

        print("\n" + "=" * 72)
        print("Done.")


def main() -> None:
    import asyncio

    asyncio.run(_demo())


if __name__ == "__main__":
    main()
