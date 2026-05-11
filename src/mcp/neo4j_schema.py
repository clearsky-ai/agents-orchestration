"""Live Neo4j graph schema introspection for MCP resources and Cypher generation."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

_SCHEMA_CACHE_TTL_SEC = 300.0
_schema_cache: dict[str, Any] = {"text": "", "expires": 0.0}
_neo4j_driver: Any | None = None

# When Neo4j is unavailable or introspection fails — keep aligned with mock loader graph.
STATIC_SCHEMA_FALLBACK = (
    "Neo4j graph schema (static fallback — live introspection unavailable):\n"
    "  (:Task {task_id, business_day, name, team, state, owner, reviewer, approver,\n"
    "          data_source, description, upstream_dependencies, completion_indicator,\n"
    "          attributes})\n"
    "  (:Evidence {evidence_id, task_id, source, summary, occurred_at})\n"
    "  (:Decision {decision_id, task_id, kind, rationale, decided_by, decided_at})\n"
    "Relationships:\n"
    "  (dependent:Task)-[:DEPENDS_ON]->(prerequisite:Task)\n"
    "  (e:Evidence)-[:ABOUT|FOR_TASK]->(t:Task)\n"
    "  (d:Decision)-[:REGARDS|FOR_TASK]->(t:Task)\n"
)

CYPHER_QUERY_INSTRUCTIONS = (
    "Query parameters: use $task_id for the central Task id. "
    "On DEPENDS_ON variable-length paths you may draft *1..$hops; the server rewrites "
    "$hops to an integer literal (Neo4j cannot parameterize quantifier bounds)."
)


def _get_neo4j_driver() -> Any | None:
    """Shared driver for schema reads (same env vars as cg_tools)."""
    global _neo4j_driver
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        return None
    if _neo4j_driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver


def _introspect_schema(session: Any) -> str:
    lines: list[str] = []
    lines.append("Neo4j graph schema (live introspection from the connected database):\n")

    labels = sorted(
        r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label AS label")
    )
    lines.append("Node labels:")
    for lab in labels:
        lines.append(f"  :{lab}")
    if not labels:
        lines.append("  (none)")

    rel_types = sorted(
        r["relationshipType"]
        for r in session.run(
            "CALL db.relationshipTypes() "
            "YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        )
    )
    lines.append("\nRelationship types:")
    for rt in rel_types:
        lines.append(f"  -[:{rt}]-")
    if not rel_types:
        lines.append("  (none)")

    # Omit propertyTypes — Neo4j emits a procedure warning that its format will change next major.
    by_label: dict[str, set[str]] = defaultdict(set)
    try:
        rows = session.run(
            """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeLabels, propertyName
            RETURN nodeLabels, propertyName
            """
        )
        for rec in rows:
            names = rec["nodeLabels"] or []
            pname = str(rec["propertyName"])
            for lab in names:
                by_label[str(lab)].add(pname)
    except Exception:
        pass

    if by_label:
        lines.append("\nNode properties (from db.schema.nodeTypeProperties):")
        for lab in sorted(by_label):
            names = sorted(by_label[lab])
            summary = ", ".join(names[:120])
            if len(names) > 120:
                summary += ", …"
            lines.append(f"  :{lab} → {summary}")

    rel_props: dict[str, set[str]] = defaultdict(set)
    try:
        rows = session.run(
            """
            CALL db.schema.relTypeProperties()
            YIELD relationshipType, propertyName
            RETURN relationshipType, propertyName
            """
        )
        for rec in rows:
            rt = str(rec["relationshipType"])
            rel_props[rt].add(str(rec["propertyName"]))
    except Exception:
        pass

    if rel_props:
        lines.append("\nRelationship properties:")
        for rt in sorted(rel_props):
            names = sorted(rel_props[rt])
            summary = ", ".join(names)
            lines.append(f"  -[:{rt}]-  {summary}")

    try:
        crows = session.run(
            "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN *"
        )
        citems = list(crows)
        if citems:
            lines.append("\nConstraints (SHOW CONSTRAINTS):")
            for r in citems[:50]:
                lines.append(
                    f"  {r.get('name')}  {r.get('type')}  {r.get('entityType')}  "
                    f"{r.get('labelsOrTypes')}  {r.get('properties')}"
                )
            if len(citems) > 50:
                lines.append("  …")
    except Exception:
        try:
            crows = session.run("CALL db.constraints()")
            citems = list(crows)
            if citems:
                lines.append("\nConstraints:")
                for r in citems[:50]:
                    lines.append(f"  {dict(r)}")
        except Exception:
            pass

    lines.append("\n" + CYPHER_QUERY_INSTRUCTIONS)
    return "\n".join(lines)


def invalidate_graph_schema_cache() -> None:
    """Clear cached schema and schema driver (e.g. after ``load_mock_neo4j`` changes the graph)."""
    global _neo4j_driver
    _schema_cache["text"] = ""
    _schema_cache["expires"] = 0.0
    if _neo4j_driver is not None:
        try:
            _neo4j_driver.close()
        except Exception:
            pass
        _neo4j_driver = None


def get_graph_schema_for_cypher(force_refresh: bool = False) -> str:
    """Return schema text for DSPy grounding and MCP resource body (cached)."""
    now = time.monotonic()
    if (
        not force_refresh
        and _schema_cache["text"]
        and now < float(_schema_cache["expires"])
    ):
        return str(_schema_cache["text"])

    driver = _get_neo4j_driver()
    if driver is None:
        text = STATIC_SCHEMA_FALLBACK + "\n\n" + CYPHER_QUERY_INSTRUCTIONS
        _schema_cache["text"] = text
        _schema_cache["expires"] = now + 60.0
        return text

    try:
        with driver.session() as session:
            text = _introspect_schema(session)
    except Exception:
        text = STATIC_SCHEMA_FALLBACK + "\n\n" + CYPHER_QUERY_INSTRUCTIONS

    _schema_cache["text"] = text
    _schema_cache["expires"] = now + _SCHEMA_CACHE_TTL_SEC
    return text
