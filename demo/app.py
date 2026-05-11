import pandas as pd
import streamlit as st

st.set_page_config(page_title="Veltara Agent Demo", layout="wide", page_icon="🔄")

# ── Mock outputs per thread ────────────────────────────────────────────────────
# Each entry maps a Thread ID to one of 3 output types: pmo_write | escalation | notification
THREAD_OUTPUTS: dict = {
    "MEC-EM0007": {
        "type": "pmo_write",
        "task_id": "T01",
        "task_name": "Accrual intake kickoff (request unbilled spend)",
        "transition_from": "IN_PROGRESS",
        "transition_to": "COMPLETE",
        "evidence": "Heather Walsh email approval — confidence 0.95",
        "authorized_by": "heather.walsh@veltara.com",
        "timestamp": "2026-01-28T08:42:00Z",
    },
    "MEC-EM0040": {
        "type": "escalation",
        "task_id": "T01",
        "task_name": "Accrual intake kickoff (request unbilled spend)",
        "proposed_transition": "IN_PROGRESS → BLOCKED",
        "routed_to": "Elena Martinez",
        "routed_to_email": "elena.martinez@veltara.com",
        "reason": "CRITICAL — AP interface batch contains 38 duplicate records inflating sub-ledger by $347K. All AP posting suspended pending reversal.",
        "risk": "HIGH",
        "confidence": "0.89",
    },
    "MEC-EM0021": {
        "type": "escalation",
        "task_id": "T05",
        "task_name": "Global Payroll Register Validation & Headcount Reconciliation",
        "proposed_transition": "IN_PROGRESS → COMPLETE",
        "routed_to": "Melissa Carter",
        "routed_to_email": "melissa.carter@veltara.com",
        "reason": "Headcount reconciliation validated (1,247/1,247). Confidence 0.91 clears threshold. Mandatory approver sign-off required per task config.",
        "risk": "MEDIUM",
        "confidence": "0.91",
    },
    "MEC-EM0054": {
        "type": "notification",
        "task_id": "T07",
        "task_name": "Validate Contract & Order Completeness",
        "to": "Fatima Ali",
        "to_email": "fatima.ali@veltara.com",
        "subject": "T07 — Additional evidence needed to complete validation",
        "priority": "MEDIUM",
        "body": (
            "The system reviewed the ClearPath Systems contract amendment thread "
            "but cannot confirm T07 completion.\n\n"
            "Missing items:\n"
            "  1. Kevin O'Donnell rule change request — please confirm submitted\n"
            "  2. Orion Analytics assessment — please attach supporting document\n"
            "  3. Explicit confirmation all contracts in scope are validated\n\n"
            "Task T07 remains IN_PROGRESS. Please reply when these items are ready."
        ),
        "confidence": "0.42",
    },
    "MEC-EM0009": {
        "type": "pmo_write",
        "task_id": "T02",
        "task_name": "Confirm payroll calendars closed (final payrolls processed)",
        "transition_from": "IN_PROGRESS",
        "transition_to": "COMPLETE",
        "evidence": "Steven Price confirmation — all regional payrolls processed — confidence 0.93",
        "authorized_by": "steven.price@veltara.com",
        "timestamp": "2026-01-28T09:30:00Z",
    },
    "MEC-EM0001": {
        "type": "notification",
        "task_id": "T01",
        "task_name": "Accrual intake kickoff (request unbilled spend)",
        "to": "Regional Close Leads",
        "to_email": "close-leads@veltara.com",
        "subject": "January 2026 Close — Kickoff signal received, monitoring T01",
        "priority": "LOW",
        "body": (
            "Close kickoff email from Elena Martinez received and logged.\n\n"
            "T01 (Accrual intake kickoff) is now IN_PROGRESS.\n"
            "Carry-over items from December flagged for BD02:\n"
            "  - APAC lease modification reclass\n"
            "  - EMEA intercompany timing difference\n\n"
            "Board package due: BD10 (one day earlier than usual)."
        ),
        "confidence": "0.78",
    },
    "MEC-EM0028": {
        "type": "escalation",
        "task_id": "T03",
        "task_name": "Billing & Revenue Cutoff Validation",
        "proposed_transition": "IN_PROGRESS → BLOCKED",
        "routed_to": "Olivia Chen",
        "routed_to_email": "olivia.chen@veltara.com",
        "reason": "Orion Analytics contract missed billing cutoff. Invoice not raised before period end. Revenue recognition at risk for January.",
        "risk": "HIGH",
        "confidence": "0.88",
    },
    "MEC-EM0049": {
        "type": "pmo_write",
        "task_id": "T11",
        "task_name": "Post Payroll Accrual JE",
        "transition_from": "IN_PROGRESS",
        "transition_to": "COMPLETE",
        "evidence": "Emily Johnson — Fixed asset depreciation run posted — confidence 0.96",
        "authorized_by": "emily.johnson@veltara.com",
        "timestamp": "2026-02-04T14:00:00Z",
    },
    "MEC-EM0066": {
        "type": "pmo_write",
        "task_id": "T14",
        "task_name": "Run Currency Revaluation",
        "transition_from": "IN_PROGRESS",
        "transition_to": "COMPLETE",
        "evidence": "Robert Chen — FX revaluation January rates applied — confidence 0.97",
        "authorized_by": "robert.chen@veltara.com",
        "timestamp": "2026-02-05T10:15:00Z",
    },
    "MEC-EM0012": {
        "type": "notification",
        "task_id": "T15",
        "task_name": "GL / Balance Sheet Reconciliation Review (NA)",
        "to": "Robert Chen, Elena Martinez",
        "to_email": "robert.chen@veltara.com",
        "subject": "NA Data Coordination — Pre-close alignment signal logged",
        "priority": "LOW",
        "body": (
            "North America pre-close coordination email from Robert Chen logged.\n\n"
            "T15 (GL / Balance Sheet Reconciliation) monitoring started.\n"
            "No completion signal detected yet — awaiting reconciliation confirmation.\n\n"
            "Task T15 remains IN_PROGRESS."
        ),
        "confidence": "0.61",
    },
}

# Fallback classification based on subject keywords
def _classify_thread(subject: str, body: str) -> dict:
    s = (subject + " " + body).lower()
    if any(w in s for w in ["critical", "error", "duplicate", "missed", "blocked", "exception", "flag"]):
        return {
            "type": "escalation",
            "task_id": "—",
            "task_name": "Unknown task",
            "proposed_transition": "IN_PROGRESS → BLOCKED",
            "routed_to": "Close Owner",
            "routed_to_email": "close-owner@veltara.com",
            "reason": "Anomaly or blocker signal detected in email thread. Manual review required.",
            "risk": "MEDIUM",
            "confidence": "0.75",
        }
    if any(w in s for w in ["approved", "complete", "confirmed", "posted", "validated", "sent", "processed", "done"]):
        return {
            "type": "pmo_write",
            "task_id": "—",
            "task_name": "Unknown task",
            "transition_from": "IN_PROGRESS",
            "transition_to": "COMPLETE",
            "evidence": "Completion signal detected in email thread — confidence 0.82",
            "authorized_by": "unknown",
            "timestamp": "2026-01-28T00:00:00Z",
        }
    return {
        "type": "notification",
        "task_id": "—",
        "task_name": "Unknown task",
        "to": "Close Team",
        "to_email": "close-team@veltara.com",
        "subject": "Signal logged — no actionable transition detected",
        "priority": "LOW",
        "body": "Email thread logged. No completion or anomaly signal detected with sufficient confidence. Monitoring continues.",
        "confidence": "0.55",
    }


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_emails() -> pd.DataFrame:
    return pd.read_csv("data/emails.csv")


emails_df = load_emails()

threads: dict = {}
for _, row in emails_df.iterrows():
    tid = row["Thread ID"]
    if tid not in threads:
        threads[tid] = {
            "subject": row["Message Subject"],
            "messages": [],
        }
    threads[tid]["messages"].append(row)


# ── Output card renderers ──────────────────────────────────────────────────────
def _field_row(label: str, value: str) -> str:
    return (
        f'<div style="display:flex;gap:0;border-bottom:1px solid #f3f4f6;padding:8px 0;">'
        f'<span style="min-width:140px;font-size:12px;color:#9ca3af;font-weight:500">{label}</span>'
        f'<span style="font-size:13px;color:#111827;flex:1">{value}</span>'
        f'</div>'
    )


def render_pmo_write(out: dict) -> None:
    st.markdown(
        f"""
<div style="border:2px solid #22c55e;border-radius:12px;overflow:hidden;">
  <div style="background:#22c55e;padding:12px 18px;display:flex;align-items:center;gap:10px;">
    <span style="font-size:22px">⚡</span>
    <span style="font-weight:700;font-size:16px;color:white">PMO Write Tool Called</span>
  </div>
  <div style="padding:16px 18px;background:white;">
    {_field_row("Task ID", out["task_id"])}
    {_field_row("Task", out["task_name"])}
    {_field_row("Transition", f'<strong>{out["transition_from"]} → {out["transition_to"]}</strong>')}
    {_field_row("Evidence", out["evidence"])}
    {_field_row("Authorized by", out["authorized_by"])}
    {_field_row("Timestamp", out["timestamp"])}
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_notification(out: dict) -> None:
    body_html = out["body"].replace("\n", "<br>")
    priority_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#6b7280"}.get(
        out.get("priority", "LOW"), "#6b7280"
    )
    st.markdown(
        f"""
<div style="border:2px solid #3b82f6;border-radius:12px;overflow:hidden;">
  <div style="background:#3b82f6;padding:12px 18px;display:flex;align-items:center;gap:10px;">
    <span style="font-size:22px">📨</span>
    <span style="font-weight:700;font-size:16px;color:white">Notification Draft Emitted</span>
  </div>
  <div style="padding:16px 18px;background:white;">
    {_field_row("Task ID", out["task_id"])}
    {_field_row("To", f'{out["to"]} &lt;{out["to_email"]}&gt;')}
    {_field_row("Subject", f'<strong>{out["subject"]}</strong>')}
    {_field_row("Priority", f'<span style="color:{priority_color};font-weight:600">{out.get("priority","—")}</span>')}
    {_field_row("Confidence", out.get("confidence", "—"))}
    <div style="margin-top:12px;background:#f9fafb;border-radius:8px;padding:12px;font-size:13px;color:#374151;line-height:1.6">
      {body_html}
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_escalation(out: dict, thread_id: str) -> None:
    risk_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(
        out.get("risk", "MEDIUM"), "#f59e0b"
    )
    st.markdown(
        f"""
<div style="border:2px solid #f59e0b;border-radius:12px;overflow:hidden;">
  <div style="background:#f59e0b;padding:12px 18px;display:flex;align-items:center;gap:10px;">
    <span style="font-size:22px">👤</span>
    <span style="font-weight:700;font-size:16px;color:white">Human Approval Required</span>
  </div>
  <div style="padding:16px 18px;background:white;">
    {_field_row("Task ID", out["task_id"])}
    {_field_row("Task", out["task_name"])}
    {_field_row("Proposed", f'<strong>{out["proposed_transition"]}</strong>')}
    {_field_row("Routed to", f'{out["routed_to"]} &lt;{out["routed_to_email"]}&gt;')}
    {_field_row("Risk", f'<span style="color:{risk_color};font-weight:600">{out.get("risk","—")}</span>')}
    {_field_row("Confidence", out.get("confidence", "—"))}
    <div style="margin-top:12px;background:#fffbeb;border-radius:8px;padding:12px;font-size:13px;color:#374151;border-left:3px solid #f59e0b;">
      {out["reason"]}
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # Approve / Reject buttons
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    approved_key = f"approved_{thread_id}"
    rejected_key = f"rejected_{thread_id}"

    with col_a:
        if st.button("✅  Approve", type="primary", use_container_width=True, key=f"btn_approve_{thread_id}"):
            st.session_state[approved_key] = True
            st.session_state.pop(rejected_key, None)
    with col_b:
        if st.button("❌  Reject", use_container_width=True, key=f"btn_reject_{thread_id}"):
            st.session_state[rejected_key] = True
            st.session_state.pop(approved_key, None)

    if st.session_state.get(approved_key):
        st.success(
            f"Approved by demo user — PMO write tool will mark **{out['task_id']}** "
            f"{out['proposed_transition'].split('→')[-1].strip()}."
        )
    elif st.session_state.get(rejected_key):
        st.error("Rejected — transition blocked. Task state unchanged.")


# ── Session state ──────────────────────────────────────────────────────────────
if "selected_thread" not in st.session_state:
    st.session_state.selected_thread = list(threads.keys())[0]
if "output" not in st.session_state:
    st.session_state.output = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Page ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex;align-items:center;gap:12px;padding-bottom:8px;">
  <span style="font-size:32px">🔄</span>
  <div>
    <h1 style="margin:0;font-size:22px;font-weight:700;color:#111827">Veltara Process Intelligence</h1>
    <p style="margin:0;color:#6b7280;font-size:13px">Month-End Close · January 2026 · Clearsky Platform</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2], gap="large")

# ── Left — thread picker + email preview ──────────────────────────────────────
with left:
    st.markdown("### Select Email Thread")

    thread_ids = list(threads.keys())
    selected_thread_id = st.selectbox(
        "Thread",
        options=thread_ids,
        format_func=lambda t: f"{t} — {threads[t]['subject'][:45]}{'…' if len(threads[t]['subject']) > 45 else ''}",
        index=thread_ids.index(st.session_state.selected_thread),
        label_visibility="collapsed",
    )

    if selected_thread_id != st.session_state.selected_thread:
        st.session_state.selected_thread = selected_thread_id
        st.session_state.output = None
        st.rerun()

    thread = threads[selected_thread_id]
    msg_count = len(thread["messages"])
    st.caption(f"{msg_count} message{'s' if msg_count > 1 else ''} in thread")

    if st.button("▶  Process with Agents", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.output = None
        st.rerun()

    st.divider()

    st.markdown("#### 📧 Emails")
    for msg in thread["messages"]:
        subj = str(msg.get("Message Subject", ""))
        label = subj[:36] + "…" if len(subj) > 36 else subj
        with st.expander(label):
            st.caption(f"**From:** {msg.get('Sender Name','')} ({msg.get('Sender Department','')})")
            st.caption(f"**To:** {msg.get('Recipient Name(s)', '')}")
            body = str(msg.get("Message Body", "")).replace("\\n", "\n")
            st.markdown(body)
            attachments = msg.get("Attachments", "")
            if attachments and str(attachments) != "nan":
                st.caption(f"📎 {attachments}")

# ── Right — agent processing + output ─────────────────────────────────────────
with right:
    st.markdown("### Output")

    output_placeholder = st.empty()

    if st.session_state.processing:
        with output_placeholder.container():
            with st.spinner("Agents processing..."):
                out = THREAD_OUTPUTS.get(selected_thread_id)
        if out is None:
            first_msg = thread["messages"][0]
            out = _classify_thread(
                str(first_msg.get("Message Subject", "")),
                str(first_msg.get("Message Body", "")),
            )

        st.session_state.output = out
        st.session_state.processing = False
        st.rerun()

    if st.session_state.output is None:
        with output_placeholder.container():
            st.markdown(
                """
<div style="border:2px dashed #e5e7eb;border-radius:12px;padding:60px 24px;text-align:center;color:#9ca3af;">
  <div style="font-size:40px;margin-bottom:12px">🔄</div>
  <div style="font-size:15px;font-weight:500">Select a thread and click <strong>Process with Agents</strong></div>
  <div style="font-size:13px;margin-top:6px">Output will appear here — PMO write, escalation, or notification</div>
</div>""",
                unsafe_allow_html=True,
            )
    else:
        out = st.session_state.output
        output_placeholder.empty()
        output_type = out["type"]

        if output_type == "pmo_write":
            render_pmo_write(out)
        elif output_type == "notification":
            render_notification(out)
        elif output_type == "escalation":
            render_escalation(out, selected_thread_id or "")
