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
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")
_CYPHER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _neo4j_driver() -> Any:
    from src.mcp import cg_tools

    driver = cg_tools._get_neo4j_driver()
    if driver is None:
        raise RuntimeError(
            "call_cg_tools requires Neo4j. Set NEO4J_PASSWORD "
            "(and optionally NEO4J_URI, NEO4J_USER), then load the graph first."
        )
    return driver


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


def _cypher_quoted_property(prop: str) -> str:
    if not _CYPHER_IDENTIFIER.fullmatch(prop):
        raise ValueError(f"Invalid property name: {prop!r}")
    return "`" + prop.replace("`", "``") + "`"


def _task_field_state(task_id: str, field: str) -> tuple[bool, Any]:
    driver = _neo4j_driver()
    with driver.session() as session:
        record = session.run(
            "MATCH (t:Task {task_id: $task_id}) RETURN t[$field] AS value, "
            "$field IN keys(t) AS exists",
            task_id=task_id,
            field=field,
        ).single()
    if record is None:
        raise RuntimeError(f"task {task_id} not found in Neo4j")
    return bool(record["exists"]), record["value"]


def _restore_task_field(task_id: str, field: str, existed: bool, value: Any) -> None:
    driver = _neo4j_driver()
    quoted_field = _cypher_quoted_property(field)
    with driver.session() as session:
        if existed:
            session.run(
                f"MATCH (t:Task {{task_id: $task_id}}) SET t.{quoted_field} = $value",
                task_id=task_id,
                value=value,
            ).consume()
        else:
            session.run(
                f"MATCH (t:Task {{task_id: $task_id}}) REMOVE t.{quoted_field}",
                task_id=task_id,
            ).consume()


def _current_task_link(node_type: str, node_id: str) -> str | None:
    driver = _neo4j_driver()
    if node_type == "Evidence":
        query = """
        MATCH (n:Evidence {evidence_id: $node_id})
        OPTIONAL MATCH (n)-[:FOR_TASK]->(t:Task)
        RETURN t.task_id AS task_id
        """
    elif node_type == "Decision":
        query = """
        MATCH (n:Decision {decision_id: $node_id})
        OPTIONAL MATCH (n)-[:FOR_TASK]->(t:Task)
        RETURN t.task_id AS task_id
        """
    else:
        raise ValueError(f"Unsupported node_type: {node_type}")
    with driver.session() as session:
        record = session.run(query, node_id=node_id).single()
    if record is None:
        raise RuntimeError(f"{node_type} {node_id} not found in Neo4j")
    return record["task_id"]


def _restore_task_link(tools: dict[str, Any], node_type: str, node_id: str, task_id: str | None) -> None:
    if task_id is None:
        return
    result = tools["link_to_task"](task_id, node_type, node_id)
    if not result.get("ok"):
        print(f"(restore {node_type} link skipped: {result})", file=sys.stderr)


def _delete_evidence(node_id: str) -> None:
    driver = _neo4j_driver()
    with driver.session() as session:
        session.run(
            "MATCH (e:Evidence {evidence_id: $id}) DETACH DELETE e",
            id=node_id,
        ).consume()


def main() -> None:
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
    field_existed, before_value = _task_field_state(tid, field)
    try:
        _print_call(
            "update_task_field",
            {"task_id": tid, "field": field, "value": "script-was-here"},
            tools["update_task_field"](tid, field, "script-was-here"),
        )
    finally:
        _restore_task_field(tid, field, field_existed, before_value)

    eid = "ev-001"
    prev_task = _current_task_link("Evidence", eid)
    try:
        _print_call(
            "link_to_task",
            {"task_id": "T01", "node_type": "Evidence", "node_id": eid},
            tools["link_to_task"]("T01", "Evidence", eid),
        )
    finally:
        _restore_task_link(tools, "Evidence", eid, prev_task)

    did = "dec-001"
    prev_dec_task = _current_task_link("Decision", did)
    try:
        _print_call(
            "link_to_task",
            {"task_id": "T01", "node_type": "Decision", "node_id": did},
            tools["link_to_task"]("T01", "Decision", did),
        )
    finally:
        _restore_task_link(tools, "Decision", did, prev_dec_task)

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
        if isinstance(eid_created, str) and eid_created:
            try:
                _delete_evidence(eid_created)
            except Exception as exc:
                print(f"(add_node cleanup skipped: {exc})", file=sys.stderr)

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
