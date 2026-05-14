"""DSPy signature and module for generating read-only Neo4j Cypher queries."""

from __future__ import annotations

import re

import dspy

# ---------------------------------------------------------------------------
# Read-only guard
# ---------------------------------------------------------------------------

_WRITE_PATTERN = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|CALL\s+apoc\.)\b",
    re.IGNORECASE,
)


def _sanitize_generated_cypher(raw: str) -> str:
    """Strip common LLM wrappers (markdown fences) so Neo4j receives bare Cypher."""
    s = raw.strip()
    s = re.sub(r"^\s*```(?:cypher|neo4j)?\s*\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()


def _cypher_for_write_scan(query: str) -> str:
    """Remove comments so words like 'set' inside // Set up... do not false-positive."""
    s = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", "", s)
    return s


def _assert_readonly(query: str) -> None:
    """Raise ValueError if the query contains any write clause (comments ignored)."""
    scan = _cypher_for_write_scan(query)
    hit = _WRITE_PATTERN.search(scan)
    if hit:
        raise ValueError(
            f"Generated query contains write keyword '{hit.group()}'; rejected."
        )


_HOPS_CAP = 30


def _inline_hops_quantifiers(cypher: str, hops: int) -> str:
    """Neo4j forbids parameters inside relationship length quantifiers (e.g. *1..$hops)."""
    n = max(1, min(int(hops), _HOPS_CAP))
    out = cypher
    out = re.sub(r"\*(\d+)\s*\.\.\s*\$hops\b", rf"*\1..{n}", out, flags=re.IGNORECASE)
    out = re.sub(r"\*\s*\.\.\s*\$hops\b", rf"*..{n}", out, flags=re.IGNORECASE)
    out = re.sub(r"\*\s*\$hops\b(?!\w)", rf"*{n}", out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class TaskContextCypherQuery(dspy.Signature):
    """Generate a read-only Neo4j Cypher MATCH query that retrieves a task and its
    N-hop DEPENDS_ON neighbourhood (both upstream prerequisites and downstream
    dependents). The query must use only MATCH, OPTIONAL MATCH, WITH, WHERE, and
    RETURN — never MERGE, CREATE, DELETE, SET, REMOVE, or DROP."""

    task_id: str = dspy.InputField(desc="task_id of the central Task node")
    hops: int = dspy.InputField(
        desc="Maximum traversal depth along DEPENDS_ON edges (typically 1–3)"
    )
    graph_schema: str = dspy.InputField(
        desc="Graph schema from the database (same text as MCP resource cp://schema/task-context)."
    )

    cypher_query: str = dspy.OutputField(
        desc=(
            "Parameterized read-only Cypher MATCH … RETURN query only — no markdown fences, "
            "no // comments. Use $task_id; you may use *1..$hops on DEPENDS_ON (bounds are "
            "inlined server-side). "
            "Must return the central task node, all reachable Task nodes within that hop depth "
            "(both directions), and the DEPENDS_ON relationships between them."
        )
    )
    reasoning: str = dspy.OutputField(
        desc="One-sentence explanation of why the query is structured this way."
    )


# ---------------------------------------------------------------------------
# DSPy Module
# ---------------------------------------------------------------------------


class TaskContextQueryGenerator(dspy.Module):
    """Generates and validates a read-only Cypher query for get_task_context."""

    def __init__(self) -> None:
        self.predict = dspy.ChainOfThought(TaskContextCypherQuery)

    def forward(self, task_id: str, hops: int = 1) -> dspy.Prediction:
        from src.mcp.neo4j_schema import get_graph_schema_for_cypher

        graph_schema = get_graph_schema_for_cypher()
        result = self.predict(task_id=task_id, hops=hops, graph_schema=graph_schema)
        clean = _sanitize_generated_cypher(result.cypher_query)
        clean = _inline_hops_quantifiers(clean, hops)
        _assert_readonly(clean)
        result.cypher_query = clean
        return result


# ---------------------------------------------------------------------------
# Convenience helper used by get_task_context
# ---------------------------------------------------------------------------

_generator: TaskContextQueryGenerator | None = None
_dspy_lm_configured: bool = False


def _ensure_dspy_lm() -> None:
    """Bind DSPy to Azure OpenAI from env (see src/common/llm/dspy.py)."""
    global _dspy_lm_configured
    if _dspy_lm_configured:
        return
    from src.common.llm.dspy import get_lm

    dspy.configure(lm=get_lm())
    _dspy_lm_configured = True


def get_task_context_query(task_id: str, hops: int = 1) -> str:
    """Return a validated read-only Cypher query for the given task + hop depth.

    Lazily initialises the module on first call so import-time DSPy
    configuration (lm, settings) is respected.
    """
    global _generator
    _ensure_dspy_lm()
    if _generator is None:
        _generator = TaskContextQueryGenerator()
    return _generator(task_id=task_id, hops=hops).cypher_query
