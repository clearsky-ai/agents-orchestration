"""MCP resources for Context Platform (live Neo4j catalog + evidence-trace semantics)."""

from __future__ import annotations


from mcp.server.fastmcp import FastMCP

# URIs used by MCP clients for resources/read.
TASK_CONTEXT_SCHEMA_URI = "cp://schema/task-context"
EVIDENCE_BUNDLE_SCHEMA_URI = "cp://schema/evidence-bundle"

# ---------------------------------------------------------------------------
# Evidence trace — semantics & payload shape (plain text; paired with live catalog)
# ---------------------------------------------------------------------------

EVIDENCE_TRACE_GUIDANCE = """
Tabular evidence row (e.g. mock_data.json evidence[])
  Each row is a timestamped communication or observation tied to a task.
  Typical keys: evidence_id (stable id), task_id (which task it concerns), source (channel,
  e.g. email, slack), summary (human-readable excerpt), occurred_at (ISO-8601, often UTC).

What evidence means
  Evidence explains context (e.g. why work is waiting). It does not by itself authorize
  overrides or status changes on a task. Use it for narrative and audit, not as a sole
  approval signal.

Thresholds & scores
  Rows in the mock bundle do not carry numeric confidence or severity. Business tolerances
  and policy calls usually live on :Decision (e.g. rationale text mentioning variance tolerance);
  those are human decisions, not fields copied onto :Evidence.

Tool API shape (get_evidence_trace)
  Responses are often wrapped as { records, out_of, page }. Each record matches the tabular
  keys above; code_query (if used) is evaluated only over those record maps in a restricted
  environment — the server does not rewrite it.

Neo4j graph projection (evidence & decisions)
  :Evidence — communications / observations; core properties evidence_id, source, summary,
  occurred_at; task_id may appear denormalized on import.
  :Decision — recorded human decisions (decision_id, task_id, kind, rationale, decided_by,
  decided_at, …).
  :Task — linked by task_id; usually matched in the graph rather than fully inlined in a trace.
  Relationship types (canonical names vary by loader):
    (:Evidence)-[:ABOUT]->(:Task)  — evidence explains work on that task.
    (:Evidence)-[:FOR_TASK]->(:Task) — alternate edge used by some loaders.
    (:Decision)-[:REGARDS]->(:Task) — decision applies to that task.
    (:Decision)-[:FOR_TASK]->(:Task) — alternate edge used by some loaders.

Illustrative Cypher (read/write sketches; align with your live schema above)
  // Evidence node + link to task
  // MERGE (e:Evidence {evidence_id: $evidence_id})
  // SET e.source = $source, e.summary = $summary, e.occurred_at = datetime($occurred_at)
  // WITH e MATCH (t:Task {task_id: $task_id}) MERGE (e)-[:ABOUT]->(t)

  // Decision node + link to task
  // MERGE (d:Decision {decision_id: $decision_id})
  // SET d += $props
  // WITH d MATCH (t:Task {task_id: $task_id}) MERGE (d)-[:REGARDS]->(t)

Data roots
  Full mock bundle: src/data/mock_data.json with tasks[], evidence[], and decisions[].
  For the live property catalog of the whole graph (including :Task), read the same MCP
  resource as task context (see URI below) — it is identical Neo4j introspection text.
""".strip()


def get_evidence_bundle_catalog() -> str:
    """Live Neo4j catalog plus evidence-trace semantics (same catalog body as task-context)."""
    from src.mcp.neo4j_schema import get_graph_schema_for_cypher

    divider = "\n\n" + "=" * 72 + "\n"
    return (
        get_graph_schema_for_cypher()
        + divider
        + "Evidence trace — semantics & payloads (plain text; catalog above is shared with "
        f"{TASK_CONTEXT_SCHEMA_URI})\n" + "=" * 72 + "\n\n" + EVIDENCE_TRACE_GUIDANCE
    )


def register_cp_resources(server: FastMCP) -> None:
    """Attach Context Platform resources (plain-text Neo4j catalog + evidence guidance)."""

    @server.resource(TASK_CONTEXT_SCHEMA_URI)
    def task_context_schema() -> str:
        """Live Neo4j graph catalog (labels, rel types, properties, constraints) for Cypher tooling."""
        from src.mcp.neo4j_schema import get_graph_schema_for_cypher

        return get_graph_schema_for_cypher()

    @server.resource(EVIDENCE_BUNDLE_SCHEMA_URI)
    def evidence_bundle_schema() -> str:
        """Live Neo4j catalog plus evidence-trace semantics (tabular shape, thresholds, tools)."""
        return get_evidence_bundle_catalog()
