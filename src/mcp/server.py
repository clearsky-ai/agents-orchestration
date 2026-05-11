import json
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP("pmo-mcp-server")

# --------------------Context Graph Tools/Resources--------------------

# --------------------Process Orchestration Tools/Resources--------------------

CURRENT_BUSINESS_DAY = "BD+2"

db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row

db.executescript(
    """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT, team TEXT, state TEXT,
    business_day TEXT, owner TEXT, description TEXT
);
CREATE TABLE activities (
    activity_id TEXT PRIMARY KEY,
    task_id TEXT, kind TEXT, actor TEXT, at TEXT
);
"""
)

DEPENDENCIES = []
ALLOWED_TASK_COLUMNS = ["name", "team", "state", "business_day", "owner", "description"]


def rows(cur):
    return [dict(row) for row in cur.fetchall()]


_data_path = Path(__file__).resolve().parent.parent / "data" / "mock_data.json"
_data = json.loads(_data_path.read_text())

for t in _data["tasks"]:
    db.execute(
        "INSERT INTO tasks (task_id, name, team, state, business_day, owner, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            t["task_id"],
            t["name"],
            t["team"],
            t["state"],
            t["business_day"],
            t["owner"],
            t["description"],
        ),
    )
    for dep in t["upstream_dependencies"]:
        DEPENDENCIES.append({"upstream": dep, "downstream": t["task_id"]})
db.commit()


@server.resource("schema://sql")
def sql_schema():
    """Full SQL schema so the agen can write valid SQL queries."""
    return "\n".join(
        r["sql"]
        for r in db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )


@server.tool()
def run_sql_query(query: str):
    """Run a read-only SELECT against the Process Orchestration store."""
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    return rows(db.execute(query))


@server.tool()
def get_task_dependencies(task_id: str):
    """Get the upstream dependencies for a given task."""
    upstream = [dep["upstream"] for dep in DEPENDENCIES if dep["downstream"] == task_id]
    downstream = [
        dep["downstream"] for dep in DEPENDENCIES if dep["upstream"] == task_id
    ]
    return {"upstream": upstream, "downstream": downstream}


@server.tool()
def process_status():
    """Today's business day, % complete, and number of in-progress tasks."""
    total = db.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    completed = db.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE state = 'complete'"
    ).fetchone()["n"]
    in_progress = db.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE state = 'in_progress'"
    ).fetchone()["n"]
    return {
        "current_business_day": CURRENT_BUSINESS_DAY,
        "percent_complete": round(completed / total * 100, 1) if total else 0,
        "in_progress_count": in_progress,
    }


@server.tool()
def update_task_attribute(task_id: str, attribute: str, value):
    """Update one attribute of one task."""
    if attribute not in ALLOWED_TASK_COLUMNS:
        raise ValueError(f"Invalid attribute: {attribute}")
    db.execute(f"UPDATE tasks SET {attribute} = ? WHERE task_id = ?", (value, task_id))
    db.commit()
    return {"task_id": task_id, "attribute": attribute, "value": value}


if __name__ == "__main__":
    server.run()
