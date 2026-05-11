import sys
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
server = FastMCP("context-graph-mcp-server")
register_cp_resources(server)
register_cg_tools(server)

if __name__ == "__main__":
    server.run()
