#!/usr/bin/env python3
"""Call every context-graph MCP tool once and print the results.

``cg_tools`` registers handlers with ``@server.tool()`` on a ``FastMCP`` instance.
This script uses a mock server whose ``.tool()`` decorator records each function,
then calls them in order (same behavior as the real server would invoke).

Run from the repo root (``.env`` in the repo root is loaded automatically)::

    docker compose up -d   # optional: local Neo4j
    PYTHONPATH=. python scripts/load_mock_neo4j.py --password "$NEO4J_PASSWORD" --data src/data/mock_data.json --clear
    PYTHONPATH=. python -m scripts.call_cg_tools
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")


def _capture_cg_tools() -> dict[str, Any]:
    from src.mcp.cg_tools import register_cg_tools

    server = MagicMock()
    captured: dict[str, Any] = {}

    def tool_decorator() -> Any:
        def deco(fn: Any) -> Any:
            captured[fn.__name__] = fn
            return fn

        return deco

    server.tool.side_effect = tool_decorator
    register_cg_tools(server)
    return captured


def _print_call(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"TOOL: {name}")
    print(f"ARGS: {json.dumps(args, default=str)}")
    out = dict(result)
    if out.get("ok") and "cypher" in out and isinstance(out["cypher"], str):
        cy = out["cypher"]
        if len(cy) > 400:
            out["cypher"] = cy[:400] + f"... ({len(cy)} chars total)"
    print("RESULT:")
    print(json.dumps(out, indent=2, default=str))


def main() -> None:
    from src.mcp import cg_tools

    tools = _capture_cg_tools()
    expected = (
        "validate_query",
        "explain_blocker",
        "find_similar_decisions",
        "get_evidence_trace",
        "get_task_context",
        "update_task_field",
        "link_to_task",
        "add_node",
    )
    missing = [n for n in expected if n not in tools]
    if missing:
        raise RuntimeError(
            f"register_cg_tools did not expose tools {missing!r}; got keys {sorted(tools)!r}"
        )

    _print_call(
        "validate_query",
        {"query": "MATCH (n) RETURN n LIMIT 1"},
        tools["validate_query"]("MATCH (n) RETURN n LIMIT 1"),
    )

    _print_call(
        "explain_blocker",
        {"task_id": "T06"},
        tools["explain_blocker"]("T06"),
    )

    _print_call(
        "find_similar_decisions",
        {"task_id": "T06", "lookback_days": 90, "limit": 5},
        tools["find_similar_decisions"]("T06", 90, 5),
    )

    _print_call(
        "get_evidence_trace",
        {"task_id": "T05", "code_query": "", "page": 1, "page_size": 10},
        tools["get_evidence_trace"]("T05", "", 1, 10),
    )

    _print_call(
        "get_task_context",
        {"task_id": "T01", "hops": 1},
        tools["get_task_context"]("T01", 1),
    )

    tid, field = "T01", "_script_demo_field"
    before = cg_tools.TASKS[tid].get(field)
    try:
        _print_call(
            "update_task_field",
            {"task_id": tid, "field": field, "value": "script-was-here"},
            tools["update_task_field"](tid, field, "script-was-here"),
        )
    finally:
        if before is None:
            cg_tools.TASKS[tid].pop(field, None)
        else:
            cg_tools.TASKS[tid][field] = before

    eid = "ev-001"
    ev = cg_tools.EVIDENCE[eid]
    prev_task = ev["task_id"]
    try:
        _print_call(
            "link_to_task",
            {"task_id": "T01", "node_type": "Evidence", "node_id": eid},
            tools["link_to_task"]("T01", "Evidence", eid),
        )
    finally:
        ev["task_id"] = prev_task

    did = "dec-001"
    dec = next((d for d in cg_tools.DECISIONS if d.get("decision_id") == did), None)
    if dec is None:
        raise RuntimeError("call_cg_tools: mock_data.json must include decision dec-001")
    prev_dec_task = dec["task_id"]
    try:
        _print_call(
            "link_to_task",
            {"task_id": "T01", "node_type": "Decision", "node_id": did},
            tools["link_to_task"]("T01", "Decision", did),
        )
    finally:
        dec["task_id"] = prev_dec_task

    add_args: dict[str, Any] = {
        "node_type": "Evidence",
        "fields": {
            "source": "call_cg_tools",
            "summary": "add_node smoke (_script_demo)",
            "occurred_at": "2026-05-12T00:00:00Z",
            "task_id": "T01",
        },
    }
    add_result = tools["add_node"](add_args["node_type"], add_args["fields"])
    _print_call("add_node", add_args, add_result)
    if add_result.get("ok") and add_args["node_type"] == "Evidence":
        eid_created = add_result.get("node_id")
        driver = cg_tools._get_neo4j_driver()
        if driver is not None and isinstance(eid_created, str) and eid_created:
            try:
                with driver.session() as session:
                    session.run(
                        "MATCH (e:Evidence {evidence_id: $id}) DETACH DELETE e",
                        id=eid_created,
                    )
            except Exception as exc:
                print(f"(add_node cleanup skipped: {exc})", file=sys.stderr)

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
