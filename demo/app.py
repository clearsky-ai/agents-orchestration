"""Veltara Agent Pipeline Demo.

Runs the real agent pipeline against scenario1_events.json and streams each
stage live into the UI as the agents emit their output.

Pre-requisites (run once before opening the app):
    docker compose up -d
    PYTHONPATH=. python scripts/load_mock_neo4j.py --clear
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import streamlit as st

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

st.set_page_config(page_title="Veltara Agent Demo", layout="wide", page_icon="🔄")

st.markdown("""
<style>
/* Tighten default Streamlit element spacing */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
/* Remove top margin on headings inside bordered containers */
[data-testid="stVerticalBlock"] h5 { margin-top: 0 !important; }
/* Give all bordered containers a cleaner look */
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #e2e8f0 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

MOCK_DATA_PATH = REPO_ROOT / "src" / "data" / "mock_data.json"

SCENARIOS: dict[str, dict] = {
    "scenario1": {
        "path": REPO_ROOT / "src" / "data" / "scenario1_events.json",
        "label": "Scenario 1 — AP Cutoff",
        "description": "Anthony Rivera → Heather Walsh · AP invoice cutoff notice",
    },
    "scenario2": {
        "path": REPO_ROOT / "src" / "data" / "scenario2_events.json",
        "label": "Scenario 2 — RSU/ESPP Reconciliation",
        "description": "Rachel Stein / Melissa Carter · Q1 equity reconciliation discrepancy",
    },
}

ANALYST_ORDER = ["process_state_analyst", "evidence_analyst", "context_research_agent"]
ANALYST_LABELS = {
    "process_state_analyst": "Process State Analyst",
    "evidence_analyst": "Evidence Analyst",
    "context_research_agent": "Context Research Agent",
}
# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_scenario(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["events"]


# ── Streaming console capture ─────────────────────────────────────────────────

class StreamingConsolePatch:
    """Redirects console.py calls into a thread-safe queue as they happen.

    Each agent call to console.section / .body / .kv etc. becomes a queue item
    of shape {"fn": <name>, "args": <tuple>}, available to the consumer in
    real time instead of only after the pipeline finishes.
    """

    _FNS = ("banner", "section", "body", "kv", "progress", "final_answer_box")

    def __init__(self, q: "queue.Queue[Any]") -> None:
        self._q = q
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> "StreamingConsolePatch":
        import src.common.console as mod
        self._mod = mod
        for fn in self._FNS:
            self._originals[fn] = getattr(mod, fn)
            setattr(mod, fn, self._make_capturer(fn))
        return self

    def __exit__(self, *_) -> None:
        for fn, orig in self._originals.items():
            setattr(self._mod, fn, orig)

    def _make_capturer(self, fn_name: str) -> Callable[..., None]:
        q = self._q
        def _fn(*args, **_kwargs) -> None:
            q.put({"fn": fn_name, "args": args})
        return _fn


# ── Stage parser ──────────────────────────────────────────────────────────────

def parse_pipeline_stages(events: list[dict]) -> dict:
    """Parse the flat captured console event list into named pipeline stages.

    Safe to call on a partial list — fields just stay None until their owning
    event arrives.
    """

    stages: dict[str, Any] = {
        "orchestrator": None,
        "analysts": {},
        "logic": {"plan": None, "policy_results": [], "deciding": False},
        "outcome": None,
        "executor": {"tools": [], "summary": None, "started": False},
    }

    state: str | None = None

    for ev in events:
        fn = ev["fn"]
        args = ev.get("args", ())

        if fn == "banner":
            title = str(args[0]) if args else ""
            if "Orchestrator" in title:
                state = "orchestrator"
                stages["orchestrator"] = {"targets": ""}

        elif fn == "section":
            title = str(args[0]) if args else ""

            if ":: task received" in title:
                agent = title.split("::")[0].strip()
                if agent in ANALYST_ORDER:
                    state = f"analyst_task:{agent}"
                    stages["analysts"].setdefault(
                        agent, {"tool_names": [], "findings": None, "started": True}
                    )
                elif agent == "executor":
                    state = "executor_task"
                    stages["executor"]["started"] = True
                else:
                    state = None

            elif ":: final reply" in title:
                agent = title.split("::")[0].strip()
                if agent in ANALYST_ORDER:
                    state = f"analyst_reply:{agent}"
                elif agent == "executor":
                    state = "executor_reply"
                else:
                    state = None

            elif "Logic :: deciding" in title:
                state = "logic_deciding"
                stages["logic"]["deciding"] = True

            elif "Logic :: action plan" in title:
                state = "logic_plan"
                stages["logic"]["plan"] = ""

            elif "Logic :: policy critique" in title:
                state = "logic_policy"

            else:
                state = None

        elif fn == "body":
            text = str(args[0]) if args else ""

            if state == "orchestrator" and stages["orchestrator"] is not None:
                stages["orchestrator"]["targets"] = text

            elif state and state.startswith("analyst_reply:"):
                agent = state.split(":")[1]
                stages["analysts"].setdefault(
                    agent, {"tool_names": [], "findings": None, "started": True}
                )
                stages["analysts"][agent]["findings"] = text

            elif state == "logic_plan":
                stages["logic"]["plan"] = (stages["logic"]["plan"] or "") + text

            elif state == "executor_reply":
                stages["executor"]["summary"] = text

        elif fn == "kv":
            label = str(args[0]) if args else ""
            value = str(args[1]) if len(args) > 1 else ""
            lower = label.lower()

            if state and state.startswith("analyst_task:"):
                agent = state.split(":")[1]
                if "tool calls" in lower or ("llm-initial" in lower and "tool" in lower):
                    for t in value.split(","):
                        t = t.strip()
                        if t:
                            stages["analysts"][agent]["tool_names"].append(t)

            elif state == "logic_policy":
                clean = label.strip()
                if clean.startswith("passed") or clean.startswith("failed"):
                    passed = clean.startswith("passed")
                    policy = clean[6:].strip()
                    stages["logic"]["policy_results"].append(
                        {"policy": policy, "passed": passed, "reason": value}
                    )

            elif state == "executor_task":
                if "tool calls" in lower or ("llm-initial" in lower and "tool" in lower):
                    for t in value.split(","):
                        t = t.strip()
                        if t:
                            stages["executor"]["tools"].append(t)

        elif fn == "final_answer_box":
            title = str(args[0]) if args else ""
            text = str(args[1]) if len(args) > 1 else ""

            if "HumanEscalation" in title:
                stages["outcome"] = {"type": "escalate", "title": title, "text": text}
            elif "no action needed" in title.lower():
                stages["outcome"] = {"type": "no_action", "title": title, "text": text}
            elif "action taken" in title:
                stages["outcome"] = {"type": "execute", "title": title, "text": text}
            elif "Logic :: decision" in title and stages["outcome"] is None:
                stages["outcome"] = {"type": "routing", "title": title, "text": text}

    return stages


# ── Graph helpers ─────────────────────────────────────────────────────────────

def check_neo4j() -> bool:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", ""),
            ),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def reload_graph() -> tuple[bool, str]:
    try:
        from neo4j import GraphDatabase
        payload = json.loads(MOCK_DATA_PATH.read_text())
        tasks = payload.get("tasks") or []
        evidence = payload.get("evidence") or []
        decisions = payload.get("decisions") or []

        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", ""),
            ),
        )
        with driver.session() as s:
            s.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            s.execute_write(lambda tx: [
                tx.run("MERGE (t:Task {task_id: $tid}) SET t = {task_id: $tid}", tid=r["task_id"])
                for r in tasks
            ])
            s.execute_write(lambda tx: [
                tx.run(
                    "MATCH (c:Task {task_id:$c}) MATCH (p:Task {task_id:$p}) MERGE (c)-[:DEPENDS_ON]->(p)",
                    c=r["task_id"], p=up,
                )
                for r in tasks for up in (r.get("upstream_dependencies") or [])
            ])
            s.execute_write(lambda tx: [
                tx.run(
                    "MERGE (e:Evidence {evidence_id:$eid}) SET e += $props "
                    "WITH e MATCH (t:Task {task_id:$tid}) MERGE (e)-[:FOR_TASK]->(t)",
                    eid=r["evidence_id"],
                    tid=r["task_id"],
                    props={k: v for k, v in r.items() if k not in ("evidence_id", "task_id")},
                )
                for r in evidence
            ])
            s.execute_write(lambda tx: [
                tx.run(
                    "MERGE (d:Decision {decision_id:$did}) SET d += $props "
                    "WITH d MATCH (t:Task {task_id:$tid}) MERGE (d)-[:FOR_TASK]->(t)",
                    did=r["decision_id"],
                    tid=r["task_id"],
                    props={k: v for k, v in r.items() if k not in ("decision_id", "task_id")},
                )
                for r in decisions
            ])
        driver.close()
        return True, f"Loaded {len(tasks)} tasks, {len(evidence)} evidence, {len(decisions)} decisions."
    except Exception as exc:
        return False, str(exc)


def _capture_cg_tools() -> dict[str, Any]:
    from src.mcp.cg_tools import register_cg_tools
    server = MagicMock()
    captured: dict[str, Any] = {}

    def _tool():
        def _deco(fn):
            captured[fn.__name__] = fn
            return fn
        return _deco

    server.tool.side_effect = _tool
    register_cg_tools(server)
    return captured


def apply_event_to_graph(event: dict) -> None:
    tools = _capture_cg_tools()
    node_type_raw = str(event.get("node_type", "")).strip().lower()
    node_type = "Evidence" if node_type_raw == "evidence" else node_type_raw.capitalize()
    fields = dict(event.get("fields") or {})

    if not fields.get("summary") and fields.get("content"):
        fields["summary"] = fields["content"]

    res = tools["add_node"](node_type, fields)
    node_id = res.get("node_id", "")
    task_id = (event.get("fields") or {}).get("task_id", "")
    if task_id and node_id:
        tools["link_to_task"](task_id, node_type, node_id)


# ── Streaming pipeline runner ─────────────────────────────────────────────────

def start_pipeline_thread(event: dict) -> tuple[str | None, "queue.Queue[Any]", threading.Event, dict]:
    """Start the pipeline in a background thread. Returns (error, q, done, error_box).

    Non-blocking — the caller must drain the queue on each Streamlit rerun.
    """
    try:
        apply_event_to_graph(event)
    except Exception as exc:
        err_q: "queue.Queue[Any]" = queue.Queue()
        err_done = threading.Event()
        err_done.set()
        return f"Graph write failed: {exc}", err_q, err_done, {"error": None}

    events_q: "queue.Queue[Any]" = queue.Queue()
    error_box: dict[str, str | None] = {"error": None}
    done = threading.Event()

    def _thread() -> None:
        try:
            async def _pipeline() -> None:
                from src.Runtime import run_pipeline_once
                from src.common.llm.azure import get_azure_lm
                lm = get_azure_lm()
                try:
                    await run_pipeline_once(
                        json.dumps(event, indent=2, ensure_ascii=False),
                        model_client=lm,
                    )
                finally:
                    await lm.close()

            with StreamingConsolePatch(events_q):
                asyncio.run(_pipeline())
        except Exception as exc:
            error_box["error"] = str(exc)
        finally:
            done.set()
            events_q.put(None)  # sentinel

    threading.Thread(target=_thread, daemon=True).start()
    return None, events_q, done, error_box


# ── Render helpers ────────────────────────────────────────────────────────────

def _initials(name: str) -> str:
    parts = name.strip().split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()


def _email_html(sender: str, recipient: str | list, source: str, task_id: str, content: str, compact: bool = False) -> str:
    if isinstance(recipient, list):
        recipient = ", ".join(recipient)
    avatar = _initials(sender) if sender else "?"
    body_safe = content.replace("\n", "<br>")
    pad = "10px 14px" if compact else "14px 18px"
    font = "12px" if compact else "13.5px"
    avatar_size = "30px" if compact else "38px"
    avatar_font = "11px" if compact else "14px"
    source_badge = f'<span style="background:#ede9fe;color:#5b21b6;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{source}</span>' if source else ""
    task_badge = f'<span style="background:#f0fdf4;color:#15803d;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{task_id}</span>' if task_id else ""
    return (
        '<div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
        f'<div style="background:#f8fafc;padding:{pad};border-bottom:1px solid #e2e8f0">'
        '<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="width:{avatar_size};height:{avatar_size};border-radius:50%;background:#e0e7ff;display:flex;align-items:center;justify-content:center;font-weight:700;color:#4338ca;font-size:{avatar_font};flex-shrink:0">{avatar}</div>'
        '<div style="flex:1;min-width:0">'
        f'<div style="font-weight:600;color:#111827;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{sender}</div>'
        f'<div style="color:#6b7280;font-size:11px;margin-top:1px">To: {recipient}</div>'
        '</div>'
        f'<div style="display:flex;gap:5px;flex-shrink:0">{source_badge}{task_badge}</div>'
        '</div>'
        '</div>'
        f'<div style="padding:{pad};font-size:{font};color:#374151;line-height:1.75;font-family:\'Georgia\',serif">{body_safe}</div>'
        '</div>'
    )


def render_event_card(event: dict) -> None:
    fields = event.get("fields") or {}
    eid = event.get("event_id", "?")
    task_id = fields.get("task_id", "")
    sender = fields.get("sender", "")
    recipient = fields.get("recipient", "")
    source = fields.get("source", "")
    content = fields.get("content", "")

    st.markdown(f"##### 📨 Event `{eid}` — Incoming Signal")
    st.markdown(
        _email_html(sender, recipient, source, task_id, content),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def render_pipeline_progress(stages: dict) -> None:
    """Compact one-line indicator of where we are in the pipeline."""
    orch_done = stages.get("orchestrator") is not None
    analysts = stages.get("analysts", {})
    analysts_done = sum(1 for a in ANALYST_ORDER if (analysts.get(a) or {}).get("findings"))
    logic = stages.get("logic", {})
    logic_done = bool(logic.get("plan")) or bool(logic.get("policy_results"))
    outcome = stages.get("outcome") or {}
    # "Decided" only turns green when executor actually reports back, not just on routing
    outcome_done = outcome.get("type") in ("execute", "escalate", "no_action")

    def chip(label: str, done: bool, active: bool) -> str:
        if done:
            return f'<span style="background:#dcfce7;color:#166534;padding:4px 12px;border-radius:20px;font-size:11.5px;font-weight:600;letter-spacing:0.01em">✓ {label}</span>'
        if active:
            return f'<span style="background:#fef9c3;color:#854d0e;padding:4px 12px;border-radius:20px;font-size:11.5px;font-weight:600;letter-spacing:0.01em">⏳ {label}</span>'
        return f'<span style="background:#f1f5f9;color:#94a3b8;padding:4px 12px;border-radius:20px;font-size:11.5px;font-weight:600;letter-spacing:0.01em">○ {label}</span>'

    parts = [
        chip("Routed", orch_done, not orch_done),
        chip(f"Analysts {analysts_done}/3", analysts_done == 3, orch_done and analysts_done < 3),
        chip("Plan", logic_done, analysts_done == 3 and not logic_done),
        chip("Decided", outcome_done, logic_done and not outcome_done),
    ]
    st.markdown(
        '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def render_analysts(stages: dict, pipeline_done: bool = False) -> None:
    """Compact summary line + collapsible expander per analyst with full findings."""
    analysts = stages.get("analysts", {})

    with st.container(border=True):
        st.markdown("##### 🔍 Analyst Findings")
        st.caption("Three specialists gathered facts in parallel · click any to expand.")

        for agent in ANALYST_ORDER:
            data = analysts.get(agent) or {}
            findings = data.get("findings")
            started = bool(data)
            errored = pipeline_done and started and not findings
            label = ANALYST_LABELS[agent]
            tool_names = data.get("tool_names") or []

            if findings:
                with st.expander(f"✅ &nbsp; {label}", expanded=False):
                    if tool_names:
                        tools_str = " · ".join(f"`{t}`" for t in tool_names)
                        st.caption(f"Tools called: {tools_str}")
                    st.markdown(findings)
            elif errored:
                with st.expander(f"⚠️ &nbsp; {label} — timed out / no response", expanded=False):
                    st.warning(
                        "This agent did not return findings — it likely timed out or hit a network error. "
                        "The pipeline continued with partial context.",
                        icon="⚠️",
                    )
            elif started:
                st.markdown(f"⏳ &nbsp; **{label}** — _running..._", unsafe_allow_html=True)
            else:
                st.markdown(f"○ &nbsp; **{label}** — _pending_", unsafe_allow_html=True)


def render_logic(stages: dict) -> None:
    """The action plan — the LogicAgent's hero output."""
    logic = stages.get("logic", {})
    plan = logic.get("plan")
    deciding = logic.get("deciding", False)
    analysts = stages.get("analysts", {})
    all_done = all((analysts.get(a) or {}).get("findings") for a in ANALYST_ORDER)

    if not plan and not deciding and not all_done:
        return  # Don't render until at least the analysts are done

    with st.container(border=True):
        st.markdown("##### 🧠 LogicAgent — Action Plan")
        if plan:
            st.markdown(plan)
        else:
            st.markdown("_⏳ Aggregating findings and generating action plan..._")


def render_policy(stages: dict) -> None:
    """Policy check — the decision point."""
    logic = stages.get("logic", {})
    plan = logic.get("plan")
    policies = logic.get("policy_results", [])

    if not plan:
        return

    with st.container(border=True):
        st.markdown("##### 📋 Policy Decision")

        if not policies:
            st.markdown("_⏳ Running policy critique..._")
            return

        for p in policies:
            icon = "✅" if p["passed"] else "❌"
            color = "#16a34a" if p["passed"] else "#dc2626"
            st.markdown(
                f'<div style="margin:8px 0"><span style="font-size:14px">{icon}</span> '
                f'<span style="font-weight:600;color:{color}">{p["policy"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="color:#64748b;font-size:12.5px;margin:-6px 0 8px 26px;line-height:1.55">{p["reason"]}</div>',
                unsafe_allow_html=True,
            )

        all_passed = all(p["passed"] for p in policies)
        if all_passed:
            st.success("**All policies passed** — routing to ExecutorAgent", icon="✅")
        else:
            st.error("**Policy failure** — escalating for human review", icon="⚠️")


def _looks_like_raw_json(text: str) -> bool:
    return "functions." in text and "returned {" in text


def _parse_executor_changes(text: str) -> list[dict]:
    """Extract structured changes from raw executor tool-return lines or markdown summaries."""
    import re
    changes = []
    # Pattern: functions.update_task_attribute returned {"task_id":..., "attribute":..., "value":...}
    for m in re.finditer(r'functions\.\w+ returned (\{[^}]+\})', text or ""):
        try:
            obj = json.loads(m.group(1))
            if "task_id" in obj and "attribute" in obj:
                changes.append(obj)
        except Exception:
            pass
    return changes


def render_outcome(stages: dict) -> None:
    """The final outcome — what actually happened."""
    outcome = stages.get("outcome")
    executor = stages.get("executor") or {}
    ex_started = executor.get("started", False)

    if not outcome:
        if ex_started:
            with st.container(border=True):
                st.markdown("##### ⚡ Executor running...")
                st.caption("Calling write tools to update the graph...")
        return

    otype = outcome.get("type", "")
    text = outcome.get("text", "")
    ex_summary = executor.get("summary", "")
    ex_tools = executor.get("tools", [])

    if otype == "execute":
        # Executor has reported back — actually done
        with st.container(border=True):
            st.markdown("##### ⚡ Execution Complete")
            changes = _parse_executor_changes(ex_summary)
            if changes:
                for c in changes:
                    task = c.get("task_id", "")
                    attr = c.get("attribute", "").replace("_", " ").title()
                    val = c.get("value", "")
                    st.markdown(
                        f'<div style="display:flex;align-items:baseline;gap:8px;padding:6px 0;border-bottom:1px solid #f1f5f9">'
                        f'<span style="color:#16a34a;font-size:15px">✓</span>'
                        f'<span style="background:#f0fdf4;color:#166534;padding:1px 7px;border-radius:8px;font-size:11px;font-weight:600;white-space:nowrap">Task {task}</span>'
                        f'<span style="color:#374151;font-size:13px">{attr}</span>'
                        f'<span style="color:#94a3b8;font-size:12px">→</span>'
                        f'<span style="color:#111827;font-size:13px;font-weight:500">{val}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            elif ex_summary and not _looks_like_raw_json(ex_summary):
                st.markdown(ex_summary)
            else:
                st.success("Actions applied to graph.", icon="✅")

    elif otype == "routing":
        # Logic routed to executor but executor hasn't reported back yet
        with st.container(border=True):
            st.markdown("##### ⚡ Executor running...")
            st.markdown("_Plan approved — executor is applying changes to the graph..._")

    elif otype == "escalate":
        with st.container(border=True):
            st.markdown("##### 👤 Human Escalation Required")
            st.warning(text)

    elif otype == "no_action":
        with st.container(border=True):
            st.markdown("##### 💤 No Action Needed")
            st.info(text)


def render_pipeline(event: dict, stages: dict, pipeline_done: bool = False) -> None:
    """Render the full pipeline trace. Handles partial stages gracefully."""
    render_event_card(event)
    render_pipeline_progress(stages)
    render_analysts(stages, pipeline_done=pipeline_done)
    render_logic(stages)
    render_policy(stages)
    render_outcome(stages)


# ── Session state ─────────────────────────────────────────────────────────────

def _init() -> None:
    if "scenario" not in st.session_state:
        st.session_state.scenario = "scenario1"
    if "results" not in st.session_state:
        st.session_state.results = {}      # scoped: {scenario_key: {eid: stages}}
    if "errors" not in st.session_state:
        st.session_state.errors = {}       # scoped: {scenario_key: {eid: error}}
    if "selected" not in st.session_state:
        st.session_state.selected = None
    if "running" not in st.session_state:
        st.session_state.running = None
    # pipeline streaming state (set while a run is active)
    if "pipe_q" not in st.session_state:
        st.session_state.pipe_q = None
    if "pipe_done" not in st.session_state:
        st.session_state.pipe_done = None
    if "pipe_error_box" not in st.session_state:
        st.session_state.pipe_error_box = None
    if "pipe_collected" not in st.session_state:
        st.session_state.pipe_collected = []


_init()

# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div style="display:flex;align-items:center;gap:12px;padding-bottom:12px;border-bottom:1px solid #e5e7eb;margin-bottom:20px">
  <span style="font-size:28px">🔄</span>
  <div>
    <h1 style="margin:0;font-size:20px;font-weight:700;color:#111827">Veltara Process Intelligence</h1>
    <p style="margin:0;color:#6b7280;font-size:12px">Agent Pipeline Demo · Month-End Close · January 2026</p>
  </div>
</div>""",
    unsafe_allow_html=True,
)

scn_key = st.session_state.scenario
scn = SCENARIOS[scn_key]
events = load_scenario(str(scn["path"]))

# Per-scenario result/error dicts
scn_results: dict = st.session_state.results.setdefault(scn_key, {})
scn_errors: dict = st.session_state.errors.setdefault(scn_key, {})

left, right = st.columns([1, 2], gap="large")

# ── Left panel ────────────────────────────────────────────────────────────────
with left:
    # Scenario picker
    chosen = st.radio(
        "Scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k]["label"],
        index=list(SCENARIOS.keys()).index(scn_key),
        horizontal=False,
        label_visibility="collapsed",
    )
    if chosen != scn_key:
        st.session_state.scenario = chosen
        st.session_state.selected = None
        st.session_state.running = None
        st.rerun()

    st.caption(scn["description"])
    st.caption(f"`{scn['path'].name}` · {len(events)} events")

    neo4j_ok = check_neo4j()
    if neo4j_ok:
        st.success("Neo4j connected", icon="✅")
    else:
        st.error("Neo4j unreachable — run `docker compose up -d`", icon="⚠️")

    col_reload, col_reset = st.columns(2)
    with col_reload:
        if st.button("↺ Reload Graph", use_container_width=True, disabled=not neo4j_ok):
            with st.spinner("Loading mock data..."):
                ok, msg = reload_graph()
            if ok:
                st.success(msg)
                st.session_state.results[scn_key] = {}
                st.session_state.errors[scn_key] = {}
            else:
                st.error(msg)
    with col_reset:
        if st.button("✖ Clear Results", use_container_width=True):
            st.session_state.results[scn_key] = {}
            st.session_state.errors[scn_key] = {}
            st.session_state.selected = None
            st.rerun()

    st.divider()
    st.markdown("**Events**")

    is_running = st.session_state.running is not None

    for event in events:
        eid = event.get("event_id", "?")
        fields = event.get("fields") or {}
        sender = fields.get("sender", "?")

        if eid in scn_errors and eid not in scn_results:
            badge, badge_color = "⚠️ Error", "#dc2626"
        elif eid in scn_results:
            otype = (scn_results[eid].get("outcome") or {}).get("type", "")
            badge, badge_color = {
                "execute": ("✅ Executed", "#16a34a"),
                "escalate": ("⚠️ Escalated", "#d97706"),
                "no_action": ("💤 No action", "#6b7280"),
                "routing": ("✅ Executed", "#16a34a"),
            }.get(otype, ("✓ Done", "#6b7280"))
        elif st.session_state.running == eid:
            badge, badge_color = "🔄 Running...", "#1d4ed8"
        else:
            badge, badge_color = "⚪ Pending", "#9ca3af"

        col_info, col_btn = st.columns([3, 2])
        with col_info:
            st.markdown(
                f"**{eid}** — {sender}<br>"
                f"<span style='font-size:11px;color:{badge_color};font-weight:600'>{badge}</span>",
                unsafe_allow_html=True,
            )
        with col_btn:
            already_run = eid in scn_results or eid in scn_errors
            btn_label = "↩ View" if already_run else "▶ Run"
            if st.button(
                btn_label,
                key=f"btn_{scn_key}_{eid}",
                use_container_width=True,
                disabled=not neo4j_ok or is_running,
            ):
                st.session_state.selected = eid
                if not already_run:
                    st.session_state.running = eid
                st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Email preview for selected event
    sel_ev = next((e for e in events if e.get("event_id") == st.session_state.selected), None)
    if sel_ev:
        st.divider()
        st.markdown("**Email**")
        sf = sel_ev.get("fields") or {}
        st.markdown(
            _email_html(
                sf.get("sender", ""),
                sf.get("recipient", ""),
                sf.get("source", ""),
                sf.get("task_id", ""),
                sf.get("content", ""),
                compact=True,
            ),
            unsafe_allow_html=True,
        )

# ── Right panel ───────────────────────────────────────────────────────────────
with right:
    running_eid = st.session_state.running

    if running_eid:
        ev = next((e for e in events if e.get("event_id") == running_eid), None)
        if ev is not None:
            # ── Start thread on first rerun for this event ──────────────────
            if st.session_state.pipe_q is None:
                start_err, pipe_q, pipe_done, pipe_error_box = start_pipeline_thread(ev)
                if start_err:
                    scn_errors[running_eid] = start_err
                    scn_results[running_eid] = parse_pipeline_stages([])
                    st.session_state.running = None
                    st.rerun()
                else:
                    st.session_state.pipe_q = pipe_q
                    st.session_state.pipe_done = pipe_done
                    st.session_state.pipe_error_box = pipe_error_box
                    st.session_state.pipe_collected = []

            # ── Drain all available queue items this rerun ──────────────────
            pipe_q = st.session_state.pipe_q
            pipe_done = st.session_state.pipe_done
            pipe_error_box = st.session_state.pipe_error_box
            collected = st.session_state.pipe_collected

            while pipe_q is not None:
                try:
                    item = pipe_q.get_nowait()
                    if item is None:
                        break
                    collected.append(item)
                except queue.Empty:
                    break

            stages = parse_pipeline_stages(collected)

            # ── Render current live state ───────────────────────────────────
            st.info(
                f"🔄 Running pipeline for **{running_eid}** — stages appear as agents finish.",
                icon="ℹ️",
            )
            render_pipeline(ev, stages, pipeline_done=False)

            # ── Check if pipeline is done ───────────────────────────────────
            if pipe_done is not None and pipe_done.is_set():
                scn_results[running_eid] = parse_pipeline_stages(collected)
                err = (pipe_error_box or {}).get("error")
                if err:
                    scn_errors[running_eid] = err
                # Clear pipeline state
                st.session_state.pipe_q = None
                st.session_state.pipe_done = None
                st.session_state.pipe_error_box = None
                st.session_state.pipe_collected = []
                st.session_state.running = None
                st.rerun()
            else:
                # Not done — come back in 0.5 s
                time.sleep(0.5)
                st.rerun()

    elif st.session_state.selected:
        eid = st.session_state.selected
        ev = next((e for e in events if e.get("event_id") == eid), None)

        if eid in scn_errors:
            st.warning(
                f"Pipeline encountered an error (partial results shown below):\n\n"
                f"`{scn_errors[eid]}`",
                icon="⚠️",
            )

        if eid in scn_results and ev:
            render_pipeline(ev, scn_results[eid], pipeline_done=True)

        else:
            st.info(f"Event {eid} selected — click **▶ Run** to execute the pipeline.")

    else:
        st.markdown(
            """
<div style="border:2px dashed #e5e7eb;border-radius:12px;padding:80px 24px;text-align:center;margin-top:20px">
  <div style="font-size:48px;margin-bottom:16px">🔄</div>
  <div style="font-size:15px;font-weight:600;color:#374151">Select an event and click <strong>▶ Run</strong></div>
  <div style="font-size:13px;color:#9ca3af;margin-top:8px">
    Each agent stage will stream in live as it finishes —<br>
    Orchestrator → Analysts → LogicAgent → PolicyCriticiser → Outcome
  </div>
</div>""",
            unsafe_allow_html=True,
        )
