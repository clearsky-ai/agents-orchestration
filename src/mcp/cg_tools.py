"""Context graph MCP tools — tasks, dependencies, evidence, decisions."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from neo4j.graph import Node, Path as Neo4jPath, Relationship

from src.mcp.cypher_generator import get_task_context_query

_neo4j_driver: Any | None = None
_CYPHER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _neo4j_not_configured(action: str) -> dict[str, Any]:
    return fail(
        "NEO4J_NOT_CONFIGURED",
        f"Set NEO4J_PASSWORD (and optionally NEO4J_URI, NEO4J_USER) to {action}.",
    )


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


def _normalize_neo4j_row_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Decode JSON strings that represent structured values while preserving plain text."""
    out = dict(props)
    for key, value in list(out.items()):
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            continue
        try:
            out[key] = json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return out


def _fetch_task(session: Any, task_id: str) -> dict[str, Any] | None:
    record = session.run(
        "MATCH (t:Task {task_id: $task_id}) RETURN t LIMIT 1",
        task_id=task_id,
    ).single()
    if record is None:
        return None
    return _normalize_neo4j_task_properties(dict(record["t"]))


def _fetch_evidence_for_task(session: Any, task_id: str) -> list[dict[str, Any]]:
    records = session.run(
        """
        MATCH (e:Evidence)-[:FOR_TASK]->(t:Task {task_id: $task_id})
        RETURN e, t.task_id AS task_id
        ORDER BY coalesce(e.occurred_at, ""), coalesce(e.evidence_id, "")
        """,
        task_id=task_id,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        row = _normalize_neo4j_row_properties(dict(record["e"]))
        row["task_id"] = record["task_id"]
        rows.append(row)
    return rows


def _fetch_decisions_for_task(
    session: Any,
    task_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    records = session.run(
        """
        MATCH (d:Decision)-[:FOR_TASK]->(t:Task {task_id: $task_id})
        RETURN d, t.task_id AS task_id
        ORDER BY coalesce(d.decided_at, ""), coalesce(d.decision_id, "")
        LIMIT $limit
        """,
        task_id=task_id,
        limit=max(0, int(limit)),
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        row = _normalize_neo4j_row_properties(dict(record["d"]))
        row["task_id"] = record["task_id"]
        rows.append(row)
    return rows


def _fetch_incomplete_upstream(session: Any, task_id: str) -> list[dict[str, Any]]:
    records = session.run(
        """
        MATCH (:Task {task_id: $task_id})-[:DEPENDS_ON]->(upstream:Task)
        WHERE coalesce(upstream.status, "") <> "complete"
        RETURN upstream
        ORDER BY upstream.task_id
        """,
        task_id=task_id,
    )
    return [
        _normalize_neo4j_task_properties(dict(record["upstream"])) for record in records
    ]


def _node_exists(
    session: Any, label: Literal["Evidence", "Decision"], node_id: str
) -> bool:
    id_prop = _ID_PROPERTY_BY_LABEL[label]
    quoted_label = _cypher_quoted_label(label)
    quoted_prop = _cypher_quoted_property(id_prop)
    record = session.run(
        f"MATCH (n:{quoted_label} {{{quoted_prop}: $node_id}}) RETURN count(n) AS count",
        node_id=node_id,
    ).single()
    return bool(record and record["count"])


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
            if downstream and upstream and (upstream, downstream) not in seen_depends:
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


def _records_to_tasks_edges(
    records: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
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


def _cypher_quoted_property(prop: str) -> str:
    """Quote a property key after validating it as a Cypher identifier."""
    if not _CYPHER_IDENTIFIER.fullmatch(prop):
        raise ValueError(f"Invalid property name: {prop!r}")
    return "`" + prop.replace("`", "``") + "`"


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
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("run the generated Cypher against Neo4j")
        try:
            with driver.session() as session:
                if _fetch_task(session, task_id) is None:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
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
        """Validate a Cypher query against Neo4j without executing writes."""
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("validate Cypher against Neo4j")
        try:
            with driver.session() as session:
                session.run(f"EXPLAIN {query}").consume()
        except Exception as e:
            return fail("INVALID_QUERY", str(e))
        return ok({"query": query, "valid": True})

    @server.tool()
    def explain_blocker(task_id: str) -> dict[str, Any]:
        """Why is the task blocked? Returns reasoning + cited evidence."""
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("read blocker context from Neo4j")
        try:
            with driver.session() as session:
                task = _fetch_task(session, task_id)
                if not task:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
                upstream = _fetch_incomplete_upstream(session, task_id)
                ev = _fetch_evidence_for_task(session, task_id)
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
        parts = []
        status = task.get("status")
        if status and status != "blocked":
            parts.append(f"Task is in status '{status}', not blocked.")
        if upstream:
            parts.append(
                "Upstream not complete: " + ", ".join(u["task_id"] for u in upstream)
            )
        attributes = task.get("attributes")
        if isinstance(attributes, dict) and "blocked_reason" in attributes:
            parts.append(f"Reason: {attributes['blocked_reason']}.")
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
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("read decisions from Neo4j")
        try:
            with driver.session() as session:
                if _fetch_task(session, task_id) is None:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
                matches = _fetch_decisions_for_task(session, task_id, limit)
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
        return ok({"decisions": matches, "count": len(matches)})

    @server.tool()
    def get_evidence_trace(
        task_id: str,
        code_query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Paginated evidence trace. code_query is opaque, server does not rewrite it."""
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("read evidence from Neo4j")
        try:
            with driver.session() as session:
                if _fetch_task(session, task_id) is None:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
                records = _fetch_evidence_for_task(session, task_id)
        except Exception as e:
            return fail("NEO4J_QUERY_FAILED", str(e))
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
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("update task fields in Neo4j")
        try:
            quoted_field = _cypher_quoted_property(field)
        except ValueError as e:
            return fail("INVALID_ARGUMENT", str(e))
        coerced = _coerce_neo4j_property(value)
        try:
            with driver.session() as session:
                task = _fetch_task(session, task_id)
                if task is None:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
                session.run(
                    f"MATCH (t:Task {{task_id: $task_id}}) SET t.{quoted_field} = $value",
                    task_id=task_id,
                    value=coerced,
                ).consume()
        except Exception as e:
            return fail("NEO4J_WRITE_FAILED", str(e))
        return ok({"task_id": task_id, "field": field, "value": value})

    @server.tool()
    def link_to_task(
        task_id: str,
        node_type: Literal["Evidence", "Decision"],
        node_id: str,
    ) -> dict[str, Any]:
        """Point an :Evidence or :Decision node at a :Task via Neo4j ``FOR_TASK``."""
        nt = node_type.strip()
        if nt not in ("Evidence", "Decision"):
            return fail(
                "INVALID_NODE_TYPE",
                "node_type must be 'Evidence' or 'Decision'.",
            )
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("link graph nodes in Neo4j")
        label = nt
        id_prop = _ID_PROPERTY_BY_LABEL[label]
        quoted_label = _cypher_quoted_label(label)
        quoted_id_prop = _cypher_quoted_property(id_prop)

        def _relink(tx: Any) -> None:
            tx.run(
                f"MATCH (n:{quoted_label} {{{quoted_id_prop}: $node_id}})-[r:FOR_TASK]->() "
                "DELETE r",
                node_id=node_id,
            )
            tx.run(
                f"""
                MATCH (t:Task {{task_id: $task_id}})
                MATCH (n:{quoted_label} {{{quoted_id_prop}: $node_id}})
                MERGE (n)-[:FOR_TASK]->(t)
                """,
                task_id=task_id,
                node_id=node_id,
            ).consume()

        try:
            with driver.session() as session:
                if _fetch_task(session, task_id) is None:
                    return fail("TASK_NOT_FOUND", f"task {task_id} not in graph")
                if not _node_exists(session, label, node_id):
                    code = (
                        "EVIDENCE_NOT_FOUND"
                        if label == "Evidence"
                        else "DECISION_NOT_FOUND"
                    )
                    return fail(code, f"{label.lower()} {node_id} not in graph")
                session.execute_write(_relink)
        except Exception as e:
            return fail("NEO4J_WRITE_FAILED", str(e))
        return ok({"task_id": task_id, "node_type": nt, "node_id": node_id})

    @server.tool()
    def add_node(
        node_type: Literal["Evidence", "Decision"],
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a Neo4j :Evidence or :Decision node with properties from ``fields``.

        ``node_type`` must be ``Evidence`` or ``Decision``. If ``evidence_id`` /
        ``decision_id`` is missing from ``fields``, a new UUID is assigned. If ``task_id`` is
        present, the new node is linked to that task via ``FOR_TASK`` in the same write.
        On success, ``node_id`` is that id property value.
        """
        driver = _get_neo4j_driver()
        if driver is None:
            return _neo4j_not_configured("create nodes in Neo4j")
        if not node_type or not node_type.strip():
            return fail("INVALID_ARGUMENT", "node_type must be a non-empty string")
        label = node_type.strip()
        if label not in ("Evidence", "Decision"):
            return fail(
                "INVALID_NODE_TYPE",
                "add_node only supports node_type 'Evidence' or 'Decision'.",
            )
        props: dict[str, Any] = {
            k: _coerce_neo4j_property(v) for k, v in (fields or {}).items()
        }
        id_prop = _ID_PROPERTY_BY_LABEL[label]
        if id_prop not in props:
            props[id_prop] = str(uuid.uuid4())
        quoted = _cypher_quoted_label(label)
        cypher = f"CREATE (n:{quoted}) SET n += $props"
        task_id = props.get("task_id")

        def _create(tx: Any) -> None:
            tx.run(cypher, props=props).consume()
            if isinstance(task_id, str) and task_id.strip():
                id_prop_quoted = _cypher_quoted_property(id_prop)
                tx.run(
                    f"""
                    MATCH (n:{quoted} {{{id_prop_quoted}: $node_id}})
                    MATCH (t:Task {{task_id: $task_id}})
                    MERGE (n)-[:FOR_TASK]->(t)
                    """,
                    node_id=props[id_prop],
                    task_id=task_id.strip(),
                ).consume()

        try:
            with driver.session() as session:
                if isinstance(task_id, str) and task_id.strip():
                    task = _fetch_task(session, task_id.strip())
                    if task is None:
                        return fail(
                            "TASK_NOT_FOUND", f"task {task_id.strip()} not in graph"
                        )
                session.execute_write(_create)
        except Exception as e:
            return fail("NEO4J_WRITE_FAILED", str(e))

        node_id = str(props[id_prop])
        return ok({"node_id": node_id})
