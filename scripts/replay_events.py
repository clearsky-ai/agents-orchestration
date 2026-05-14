#!/usr/bin/env python3
"""Replay a scenario `events.json`: apply each event to Neo4j, then run the agent pipeline.

For each event: (1) update the graph via context-graph MCP handlers (in-process),
(2) start a **fresh** ``SingleThreadedAgentRuntime``, publish the event as JSON text
to the orchestrator, and wait until idle.

Run from the repo root (``.env`` in the repo root is loaded automatically)::

    docker compose up -d   # local Neo4j
    PYTHONPATH=. python scripts/load_mock_neo4j.py --password "$NEO4J_PASSWORD" --data src/data/mock_data.json --clear
    PYTHONPATH=. python scripts/replay_events.py --events src/data/scenario1_events.json

``NEO4J_PASSWORD`` is required (same as ``add_node`` / ``link_to_task`` writes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")
DEFAULT_OUTPUT = REPO_ROOT / "src" / "data" / "mock_data.replayed.json"


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


def _open_neo4j_driver() -> Any:
    if not os.environ.get("NEO4J_PASSWORD"):
        print(
            "replay_events: NEO4J_PASSWORD is not set. Graph writes require Neo4j.\n"
            "Example: docker compose up -d && export NEO4J_PASSWORD=...\n"
            "Then load the graph: PYTHONPATH=. python scripts/load_mock_neo4j.py "
            '--password "$NEO4J_PASSWORD" --data src/data/mock_data.json --clear',
            file=sys.stderr,
        )
        sys.exit(1)
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))


def _verify_neo4j_configured() -> None:
    driver = _open_neo4j_driver()
    try:
        driver.verify_connectivity()
    except Exception as exc:
        print(f"replay_events: Neo4j connectivity check failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.close()


def _decode_json_property(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _node_props(node: Any) -> dict[str, Any]:
    return {key: _decode_json_property(value) for key, value in dict(node).items()}


def _export_graph_snapshot(output_path: Path) -> None:
    driver = _open_neo4j_driver()
    try:
        with driver.session() as session:
            task_records = session.run("""
                MATCH (t:Task)
                OPTIONAL MATCH (t)-[:DEPENDS_ON]->(upstream:Task)
                RETURN t, collect(upstream.task_id) AS upstream_dependencies
                ORDER BY t.task_id
                """)
            tasks: list[dict[str, Any]] = []
            for record in task_records:
                row = {"task_id": record["t"]["task_id"]}
                upstream = [
                    task_id
                    for task_id in record["upstream_dependencies"]
                    if isinstance(task_id, str)
                ]
                row["upstream_dependencies"] = sorted(upstream)
                tasks.append(row)

            evidence_records = session.run("""
                MATCH (e:Evidence)
                OPTIONAL MATCH (e)-[:FOR_TASK]->(t:Task)
                RETURN e, t.task_id AS task_id
                ORDER BY coalesce(e.occurred_at, ""), coalesce(e.evidence_id, "")
                """)
            evidence: list[dict[str, Any]] = []
            for record in evidence_records:
                row = _node_props(record["e"])
                linked_task_id = record["task_id"]
                row["task_id"] = linked_task_id or row.get("task_id", "")
                evidence.append(row)

            decision_records = session.run("""
                MATCH (d:Decision)
                OPTIONAL MATCH (d)-[:FOR_TASK]->(t:Task)
                RETURN d, t.task_id AS task_id
                ORDER BY coalesce(d.decided_at, ""), coalesce(d.decision_id, "")
                """)
            decisions: list[dict[str, Any]] = []
            for record in decision_records:
                row = _node_props(record["d"])
                linked_task_id = record["task_id"]
                row["task_id"] = linked_task_id or row.get("task_id", "")
                decisions.append(row)
    finally:
        driver.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"tasks": tasks, "evidence": evidence, "decisions": decisions},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nReplay snapshot written to {output_path}")


def _normalize_node_type(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r == "evidence":
        return "Evidence"
    if r == "decision":
        return "Decision"
    return raw.strip()


def _apply_event_to_graph(event: dict[str, Any], tools: dict[str, Any]) -> None:
    event_id = event.get("event_id", "?")
    event_type = str(event.get("event_type", "")).lower()
    node_type = _normalize_node_type(str(event.get("node_type", "")))
    fields = event.get("fields")
    if not isinstance(fields, dict):
        print(
            f"replay_events: event {event_id!r} missing dict 'fields'",
            file=sys.stderr,
        )
        sys.exit(1)

    if event_type == "create" and node_type == "Evidence":
        add_fields = dict(fields)
        if not add_fields.get("summary") and add_fields.get("content"):
            add_fields["summary"] = add_fields["content"]

        res = tools["add_node"]("Evidence", add_fields)
        if not res.get("ok"):
            err = res.get("error", res)
            print(
                f"replay_events: add_node failed for event {event_id!r}: {err}",
                file=sys.stderr,
            )
            sys.exit(1)
        node_id = res.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            print(
                f"replay_events: add_node missing node_id for event {event_id!r}",
                file=sys.stderr,
            )
            sys.exit(1)

        task_id = fields.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            link = tools["link_to_task"](task_id.strip(), "Evidence", node_id)
            if not link.get("ok"):
                err = link.get("error", link)
                print(
                    f"replay_events: link_to_task failed for event {event_id!r}: {err}",
                    file=sys.stderr,
                )
                sys.exit(1)
        return

    print(
        f"replay_events: unsupported event "
        f"event_type={event.get('event_type')!r} node_type={event.get('node_type')!r} "
        f"(event_id={event_id!r})",
        file=sys.stderr,
    )
    sys.exit(1)


async def _replay_all(events_path: Path, output_path: Path) -> None:
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        print("replay_events: JSON must contain an 'events' array", file=sys.stderr)
        sys.exit(1)

    _verify_neo4j_configured()
    tools = _capture_cg_tools()

    from src.Runtime import run_pipeline_once
    from src.common.llm.azure import get_azure_lm

    azure_llm = get_azure_lm()
    try:
        for event in raw_events:
            if not isinstance(event, dict):
                print("replay_events: each event must be an object", file=sys.stderr)
                sys.exit(1)
            eid = event.get("event_id", "?")
            print("\n" + "=" * 72)
            print(f"Event {eid} :: step 1 — graph")
            _apply_event_to_graph(event, tools)

            print(f"Event {eid} :: step 2 — agent runtime")
            event_text = json.dumps(event, indent=2, ensure_ascii=False)
            await run_pipeline_once(event_text, model_client=azure_llm)
    finally:
        await azure_llm.close()

    _export_graph_snapshot(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to events JSON (top-level key 'events': array of objects)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the replayed graph JSON snapshot (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    path = args.events
    if not path.is_file():
        print(f"replay_events: not a file: {path}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(_replay_all(path.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
