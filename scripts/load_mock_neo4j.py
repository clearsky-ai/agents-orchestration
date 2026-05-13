#!/usr/bin/env python3
"""Load mock_data.json into Neo4j as Task, Evidence, and Decision nodes.

Task nodes intentionally store only ``task_id``. Other task fields from the
source JSON are used only for reconstructing relationships, not node properties.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_DATA = REPO_ROOT / "src" / "data" / "mock_data.json"

CONSTRAINTS = """
CREATE CONSTRAINT task_id_unique IF NOT EXISTS
FOR (t:Task) REQUIRE t.task_id IS UNIQUE;
CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS
FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE;
CREATE CONSTRAINT decision_id_unique IF NOT EXISTS
FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE;
"""


def _value_for_neo4j(value):
    """Neo4j properties must be primitives or homogeneous lists of primitives."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, list):
        if value and not all(
            isinstance(x, (str, int, float, bool)) or x is None for x in value
        ):
            return json.dumps(value)
        return value
    return str(value)


def _row_props(record: dict, exclude: set[str]) -> dict:
    return {k: _value_for_neo4j(v) for k, v in record.items() if k not in exclude}


def ensure_constraints(tx) -> None:
    for stmt in CONSTRAINTS.strip().split(";"):
        s = stmt.strip()
        if s:
            tx.run(s)


def load_tasks(tx, tasks: list[dict]) -> None:
    for row in tasks:
        tid = row["task_id"]
        tx.run(
            """
            MERGE (t:Task {task_id: $task_id})
            SET t = {task_id: $task_id}
            """,
            task_id=tid,
        )


def load_task_dependencies(tx, tasks: list[dict]) -> None:
    for row in tasks:
        child_id = row["task_id"]
        for upstream_id in row.get("upstream_dependencies") or []:
            tx.run(
                """
                MATCH (child:Task {task_id: $child_id})
                MATCH (parent:Task {task_id: $parent_id})
                MERGE (child)-[:DEPENDS_ON]->(parent)
                """,
                child_id=child_id,
                parent_id=upstream_id,
            )


def load_evidence(tx, items: list[dict]) -> None:
    for row in items:
        eid = row["evidence_id"]
        tid = row["task_id"]
        props = _row_props(row, {"evidence_id", "task_id"})
        tx.run(
            """
            MERGE (e:Evidence {evidence_id: $evidence_id})
            SET e += $props
            WITH e
            MATCH (t:Task {task_id: $task_id})
            MERGE (e)-[:FOR_TASK]->(t)
            """,
            evidence_id=eid,
            task_id=tid,
            props=props,
        )


def load_decisions(tx, items: list[dict]) -> None:
    for row in items:
        did = row["decision_id"]
        tid = row["task_id"]
        props = _row_props(row, {"decision_id", "task_id"})
        tx.run(
            """
            MERGE (d:Decision {decision_id: $decision_id})
            SET d += $props
            WITH d
            MATCH (t:Task {task_id: $task_id})
            MERGE (d)-[:FOR_TASK]->(t)
            """,
            decision_id=did,
            task_id=tid,
            props=props,
        )


def clear_graph(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Bolt URI (default: env NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j user (default: env NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password (default: env NEO4J_PASSWORD)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Path to mock_data.json (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all nodes and relationships before load",
    )
    args = parser.parse_args()

    if not args.password:
        parser.error("Set --password or NEO4J_PASSWORD")

    payload = json.loads(args.data.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") or []
    evidence = payload.get("evidence") or []
    decisions = payload.get("decisions") or []

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            session.execute_write(ensure_constraints)
            if args.clear:
                session.execute_write(clear_graph)
            session.execute_write(load_tasks, tasks)
            session.execute_write(load_task_dependencies, tasks)
            session.execute_write(load_evidence, evidence)
            session.execute_write(load_decisions, decisions)
    finally:
        driver.close()

    try:
        from src.mcp.neo4j_schema import invalidate_graph_schema_cache

        invalidate_graph_schema_cache()
    except Exception:
        pass

    print(
        f"Loaded {len(tasks)} Task, {len(evidence)} Evidence, {len(decisions)} Decision nodes "
        f"from {args.data}"
    )


if __name__ == "__main__":
    main()
