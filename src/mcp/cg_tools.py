"""Context graph MCP tools — tasks, dependencies, evidence, decisions."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from neo4j.graph import Node, Path as Neo4jPath, Relationship

from src.mcp.cypher_generator import get_task_context_query

REPO_ROOT = Path(__file__).resolve().parents[2]
_MOCK_PATH = REPO_ROOT / "src" / "data" / "mock_data.json"


def _load_mock_graph() -> None:
    global TASKS, EDGES, EVIDENCE, DECISIONS
    payload = json.loads(_MOCK_PATH.read_text(encoding="utf-8"))
    TASKS = {t["task_id"]: t for t in payload["tasks"]}
    EDGES = []
    for t in payload["tasks"]:
        tid = t["task_id"]
        for up in t.get("upstream_dependencies") or []:
            EDGES.append({"upstream": up, "downstream": tid})
    EVIDENCE = {e["evidence_id"]: e for e in payload.get("evidence") or []}
    DECISIONS = list(payload.get("decisions") or [])


TASKS: dict[str, dict[str, Any]]
EDGES: list[dict[str, str]]
EVIDENCE: dict[str, dict[str, Any]]
DECISIONS: list[dict[str, Any]]

_load_mock_graph()

_neo4j_driver: Any | None = None


def ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **result}


def fail(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _get_neo4j_driver() -> Any | None:
    """Return a shared driver if NEO4J_PASSWORD is set; otherwise None."""
    global _neo4j_driver
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        return None
    if _neo4j_driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver


def _normalize_neo4j_task_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON-encoded ``attributes`` produced by scripts/load_mock_neo4j.py."""
    out = dict(props)
    raw = out.get("attributes")
    if isinstance(raw, str):
        try:
            out["attributes"] = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            out["attributes"] = {}
    return out


def _collect_graph_value(
    val: Any,
    tasks_map: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    seen_depends: set[tuple[str, str]],
) -> None:
    if isinstance(val, Node):
        if "Task" in val.labels and val.get("task_id") is not None:
            tasks_map[val["task_id"]] = _normalize_neo4j_task_properties(dict(val))
        return
    if isinstance(val, Relationship) and val.type == "DEPENDS_ON":
        start, end = val.start_node, val.end_node
        if "Task" in start.labels and "Task" in end.labels:
            downstream = start.get("task_id")
            upstream = end.get("task_id")
            if (
                downstream
                and upstream
                and (upstream, downstream) not in seen_depends
            ):
                seen_depends.add((upstream, downstream))
                edges.append({"upstream": upstream, "downstream": downstream})
        return
    if isinstance(val, Neo4jPath):
        for n in val.nodes:
            _collect_graph_value(n, tasks_map, edges, seen_depends)
        for r in val.relationships:
            _collect_graph_value(r, tasks_map, edges, seen_depends)
        return
    if isinstance(val, (list, tuple)):
        for x in val:
            _collect_graph_value(x, tasks_map, edges, seen_depends)


def _records_to_tasks_edges(records: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    tasks_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        for v in record.values():
            _collect_graph_value(v, tasks_map, edges, seen)
    return list(tasks_map.values()), edges


def _coerce_neo4j_property(value: Any) -> Any:
    """Map Python values to Neo4j-supported property types (primitives or JSON strings)."""
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


def _cypher_quoted_label(label: str) -> str:
    """Quote a label for Cypher; ``label`` must already be allow-listed (e.g. from db.labels())."""
    return "`" + label.replace("`", "``") + "`"


# Primary id property per label (aligns with scripts/load_mock_neo4j.py constraints).
_ID_PROPERTY_BY_LABEL: dict[str, str] = {
    "Task": "task_id",
    "Evidence": "evidence_id",
    "Decision": "decision_id",
}


def register_cg_tools(server: FastMCP) -> None:
    @server.tool()
    def get_task_context(task_id: str, hops: int = 1) -> dict[str, Any]:
        """Task + N-hop dependency neighborhood via generated read-only Cypher on Neo4j."""
        if task_id not in TASKS:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        driver = _get_neo4j_driver()
        if driver is None:
            return fail(
                "NEO4J_NOT_CONFIGURED",
                "Set NEO4J_PASSWORD (and optionally NEO4J_URI, NEO4J_USER) to run the "
                "generated Cypher against Neo4j.",
            )
        try:
            cypher = get_task_context_query(task_id, hops)
        except ValueError as e:
            return fail("INVALID_QUERY", str(e))
        except Exception as e:
            return fail("CYPHER_GENERATION_FAILED", str(e))
        try:
            with driver.session() as session:
                result = session.run(cypher, task_id=task_id, hops=hops)
                records = list(result)
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
        tasks, edges = _records_to_tasks_edges(records)
        if not tasks:
            return fail(
                "NEO4J_EMPTY_RESULT",
                "Query returned no :Task nodes. Ensure the graph is loaded "
                "(e.g. scripts/load_mock_neo4j.py) and the generated Cypher matches the schema.",
            )
        return ok({"tasks": tasks, "edges": edges, "cypher": cypher})

    @server.tool()
    def validate_query(query: str) -> dict[str, Any]:
        """Validate a query."""
        return ok({"query": query, "valid": True})

    @server.tool()
    def explain_blocker(task_id: str) -> dict[str, Any]:
        """Why is the task blocked? Returns reasoning + cited evidence."""
        task = TASKS.get(task_id)
        if not task:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        upstream = [
            TASKS[e["upstream"]]
            for e in EDGES
            if e["downstream"] == task_id
            and TASKS[e["upstream"]]["status"] != "complete"
        ]
        ev = [e for e in EVIDENCE.values() if e["task_id"] == task_id]
        parts = []
        if task["status"] != "blocked":
            parts.append(f"Task is in status '{task['status']}', not blocked.")
        if upstream:
            parts.append(
                "Upstream not complete: " + ", ".join(u["task_id"] for u in upstream)
            )
        if "blocked_reason" in task["attributes"]:
            parts.append(f"Reason: {task['attributes']['blocked_reason']}.")
        return ok(
            {
                "task_id": task_id,
                "reasoning": " ".join(parts) or "No blocker found.",
                "evidence": ev,
                "upstream_blockers": upstream,
            }
        )

    @server.tool()
    def find_similar_decisions(
        task_id: str, lookback_days: int = 90, limit: int = 10
    ) -> dict[str, Any]:
        """Past decisions on this task within the lookback window."""
        if task_id not in TASKS:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        matches = [d for d in DECISIONS if d["task_id"] == task_id][:limit]
        return ok({"decisions": matches, "count": len(matches)})

    @server.tool()
    def get_evidence_trace(
        task_id: str,
        code_query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Paginated evidence trace. code_query is opaque, server does not rewrite it."""
        if task_id not in TASKS:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        records = [e for e in EVIDENCE.values() if e["task_id"] == task_id]
        if code_query:
            try:
                records = [
                    r for r in records if eval(code_query, {"__builtins__": {}}, r)
                ]
            except Exception as e:
                return fail("INVALID_ARGUMENT", f"code_query: {e}")
        total = len(records)
        start = (page - 1) * page_size
        if start >= total and total > 0:
            return fail("PAGINATION_EXHAUSTED", "no more records")
        return ok(
            {
                "records": records[start : start + page_size],
                "out_of": total,
                "page": page,
            }
        )

    @server.tool()
    def update_task_field(task_id: str, field: str, value: str) -> dict[str, Any]:
        """Update a task field. Task must exist."""
        if task_id not in TASKS:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        TASKS[task_id][field] = value
        return ok({"task_id": task_id, "field": field, "value": value})

    @server.tool()
    def link_to_task(
        task_id: str,
        node_type: Literal["Evidence", "Decision"],
        node_id: str,
    ) -> dict[str, Any]:
        """Point an :Evidence or :Decision node at a :Task (mock ``task_id`` + Neo4j ``FOR_TASK``)."""
        if task_id not in TASKS:
            return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        nt = node_type.strip()
        if nt not in ("Evidence", "Decision"):
            return fail(
                "INVALID_NODE_TYPE",
                "node_type must be 'Evidence' or 'Decision'.",
            )
        if nt == "Evidence":
            if node_id not in EVIDENCE:
                return fail("EVIDENCE_NOT_FOUND", f"evidence {node_id} not in graph")
        else:
            if not any(d.get("decision_id") == node_id for d in DECISIONS):
                return fail("DECISION_NOT_FOUND", f"decision {node_id} not in graph")

        driver = _get_neo4j_driver()
        if driver is not None:

            def _relink(tx: Any) -> None:
                if nt == "Evidence":
                    tx.run(
                        "MATCH (:Evidence {evidence_id: $node_id})-[r:FOR_TASK]->() DELETE r",
                        node_id=node_id,
                    )
                    tx.run(
                        """
                        MATCH (t:Task {task_id: $task_id})
                        MATCH (e:Evidence {evidence_id: $node_id})
                        MERGE (e)-[:FOR_TASK]->(t)
                        """,
                        task_id=task_id,
                        node_id=node_id,
                    )
                else:
                    tx.run(
                        "MATCH (:Decision {decision_id: $node_id})-[r:FOR_TASK]->() DELETE r",
                        node_id=node_id,
                    )
                    tx.run(
                        """
                        MATCH (t:Task {task_id: $task_id})
                        MATCH (d:Decision {decision_id: $node_id})
                        MERGE (d)-[:FOR_TASK]->(t)
                        """,
                        task_id=task_id,
                        node_id=node_id,
                    )

            try:
                with driver.session() as session:
                    session.execute_write(_relink)
            except Exception as e:
                return fail("NEO4J_WRITE_FAILED", str(e))

        if nt == "Evidence":
            EVIDENCE[node_id]["task_id"] = task_id
        else:
            for row in DECISIONS:
                if row.get("decision_id") == node_id:
                    row["task_id"] = task_id
                    break
        return ok({"task_id": task_id, "node_type": nt, "node_id": node_id})

    @server.tool()
    def add_node(node_type: Literal["Evidence", "Decision"], fields: dict[str, Any]) -> dict[str, Any]:
        """Create a Neo4j :Evidence or :Decision node with properties from ``fields``.

        ``node_type`` must be ``Evidence`` or ``Decision``. The label must also exist in the
        database (``CALL db.labels()``). If ``evidence_id`` / ``decision_id`` is missing from
        ``fields``, a new UUID is assigned. On success, ``node_id`` is that id property value.
        """
        driver = _get_neo4j_driver()
        if driver is None:
            return fail(
                "NEO4J_NOT_CONFIGURED",
                "Set NEO4J_PASSWORD (and optionally NEO4J_URI, NEO4J_USER) to create nodes "
                "in Neo4j.",
            )
        if not node_type or not node_type.strip():
            return fail("INVALID_ARGUMENT", "node_type must be a non-empty string")
        label = node_type.strip()
        if label not in ("Evidence", "Decision"):
            return fail(
                "INVALID_NODE_TYPE",
                "add_node only supports node_type 'Evidence' or 'Decision'.",
            )
        try:
            with driver.session() as session:
                rows = list(
                    session.run(
                        "CALL db.labels() YIELD label RETURN label AS label ORDER BY label"
                    )
                )
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
        existing = {str(r["label"]) for r in rows}
        if label not in existing:
            known = ", ".join(sorted(existing)) if existing else "(none)"
            return fail(
                "UNKNOWN_NODE_TYPE",
                f"Label {label!r} is not present in the database. Existing labels: {known}.",
            )
        props: dict[str, Any] = {
            k: _coerce_neo4j_property(v) for k, v in (fields or {}).items()
        }
        id_prop = _ID_PROPERTY_BY_LABEL.get(label)
        if id_prop is not None and id_prop not in props:
            props[id_prop] = str(uuid.uuid4())
        quoted = _cypher_quoted_label(label)
        cypher = f"CREATE (n:{quoted}) SET n += $props"

        def _create(tx: Any) -> None:
            tx.run(cypher, props=props)

        try:
            with driver.session() as session:
                session.execute_write(_create)
        except Exception as e:
            return fail("NEO4J_WRITE_FAILED", str(e))
        return ok({"node_id": str(props[id_prop])})