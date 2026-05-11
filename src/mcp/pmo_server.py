import json
import sys
import sqlite3
from pathlib import Path

# Repo root must be on sys.path so `from src...` works when this file is run
# as a script (e.g. `python src/mcp/server.py`); Python otherwise only adds `src/mcp`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from src.mcp.cg_tools import register_cg_tools
from src.mcp.cp_resources import register_cp_resources

load_dotenv()
server = FastMCP("pmo-mcp-server")

# --------------------Context Graph Tools/Resources--------------------

# --------------------Process Orchestration Tools/Resources--------------------

CURRENT_BUSINESS_DAY = "BD+2"

db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row

db.executescript("""
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT, team TEXT, state TEXT,
    business_day TEXT, owner TEXT, description TEXT
);
CREATE TABLE activities (
    activity_id TEXT PRIMARY KEY,
    task_id TEXT, kind TEXT, actor TEXT, at TEXT
);
""")

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
        (t["task_id"], t["name"], t["team"], t["state"],
         t["business_day"], t["owner"], t["description"]),
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


@server.tool(
    name="run_sql_query",
    description=(
        "Execute a read-only SELECT against the in-memory Process Orchestration database "
        "(tasks, activities). Use the schema://sql resource for table DDL. "
        "Non-SELECT statements are rejected."
    ),
)
def run_sql_query(query: str):
    """Run a read-only SELECT against the Process Orchestration store."""
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    return rows(db.execute(query))


@server.tool(
    name="get_task_dependencies",
    description=(
        "Return dependency edges for a task: upstream task IDs that must complete first, "
        "and downstream task IDs that depend on this task. Dependency data is loaded from mock seed data."
    ),
)
def get_task_dependencies(task_id: str):
    """Get the upstream dependencies for a given task."""
    upstream = [dep["upstream"] for dep in DEPENDENCIES if dep["downstream"] == task_id]
    downstream = [
        dep["downstream"] for dep in DEPENDENCIES if dep["upstream"] == task_id
    ]
    return {"upstream": upstream, "downstream": downstream}


@server.tool(
    name="process_status",
    description=(
        "High-level portfolio snapshot: configured current business day (BD offset), "
        "percent of tasks in complete state, and count of tasks in in_progress state."
    ),
)
def process_status():
    """Today's business day, % complete, and number of in-progress tasks."""
    total = db.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    completed = db.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE state = 'complete'").fetchone()["n"]
    in_progress = db.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE state = 'in_progress'").fetchone()["n"]
    return {
        "current_business_day": CURRENT_BUSINESS_DAY,
        "percent_complete": round(completed / total * 100, 1) if total else 0,
        "in_progress_count": in_progress,
    }


@server.tool(
    name="update_task_attribute",
    description=(
        "Persist a single column update on one task by task_id. Allowed attributes: name, team, "
        "state, business_day, owner, description. Commits immediately to the in-memory store."
    ),
)
def update_task_attribute(task_id: str, attribute: str, value):
    """Update one attribute of one task."""
    if attribute not in ALLOWED_TASK_COLUMNS:
        raise ValueError(f"Invalid attribute: {attribute}")
    db.execute(f"UPDATE tasks SET {attribute} = ? WHERE task_id = ?", (value, task_id))
    db.commit()
    return {"task_id": task_id, "attribute": attribute, "value": value}


if __name__ == "__main__":
    server.run()
