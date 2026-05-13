from __future__ import annotations

import asyncio
from typing import List

import dspy
from pydantic import BaseModel

from src.common.llm.dspy import get_lm


# --- Configure DSPy's LM once on import -----------------------------------
# This module is the only DSPy consumer for now, so configuring at import
# time is the simplest scope. If a second module starts using DSPy with
# different settings, lift this into a shared init.
dspy.configure(lm=get_lm())


# --- Policy catalog -------------------------------------------------------
# Verbatim policy statements, keyed by policy family and then by the state
# transition they govern. The DSPy module routes each proposed step to the
# matching transition bucket and evaluates only those policies.
POLICIES: dict[str, dict[str, list[str]]] = {
    "task_transition_policy": {
        "ready_to_in_progress": [
            # "Done by the owner",
            "upstream dependencies are done",
        ],
        "in_progress_to_complete": [
            # "Done by the owner",
            "Task approval is received",
        ],
    },
}


# --- Structured verdict shape ---------------------------------------------
class CritiqueResult(BaseModel):
    """One verdict per policy. Rendered as a passed/failed line in the LogicAgent."""

    policy: str
    passed: bool
    reason: str


# --- DSPy signature -------------------------------------------------------
class PolicyCritique(dspy.Signature):
    """Audit a proposed action plan against a catalog of policies.

    The policies catalog is a nested dict:
      { policy_family: { state_transition: [policy_statement, ...] } }
    For example, `task_transition_policy.ready_to_in_progress` lists the
    policies that apply when a task moves from `ready` to `in_progress`.

    Steps:
      1. Read the proposed_actions and identify which state transition(s)
         each step represents (e.g. `ready_to_in_progress`).
      2. For each step, evaluate ONLY the policies listed under that
         transition. Do not evaluate policies that don't apply.
      3. Emit one CritiqueResult per (step, policy) pair you evaluate:
           - policy: echo the policy statement verbatim
           - passed: true if the plan clearly satisfies the policy, false otherwise
           - reason: one or two sentences explaining the verdict, citing
             task_ids, evidence_ids, decision_ids, owners, etc. verbatim
             from the findings.

    Be strict. If the findings do not contain enough evidence to verify a
    policy holds, mark it failed and say what evidence is missing. Do not
    invent identifiers.
    """

    analyst_findings: str = dspy.InputField(
        desc=(
            "Aggregated findings from ProcessStateAnalyst, EvidenceAnalyst, "
            "and ContextResearchAgent. The only source of facts."
        ),
    )
    proposed_actions: str = dspy.InputField(
        desc=(
            "The LogicAgent's action plan — either a list of tool calls with "
            "arguments, or a statement that no action is needed."
        ),
    )
    policies: dict[str, dict[str, list[str]]] = dspy.InputField(
        desc=(
            "Nested catalog: outer key is the policy family, inner key is "
            "the state transition, value is the list of policy statements "
            "to evaluate when that transition is proposed."
        ),
    )
    results: list[CritiqueResult] = dspy.OutputField(
        desc=(
            "One verdict per (step, policy) pair evaluated. Only include "
            "policies that apply to a transition present in the plan."
        ),
    )


# --- DSPy module ----------------------------------------------------------
class _PolicyCriticiserModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought makes the LLM reason step-by-step before producing
        # the structured verdict. For compliance-style checks the reasoning
        # trace materially improves correctness.
        self.predict = dspy.ChainOfThought(PolicyCritique)

    def forward(
        self,
        analyst_findings: str,
        proposed_actions: str,
        policies: dict[str, dict[str, list[str]]],
    ) -> dspy.Prediction:
        return self.predict(
            analyst_findings=analyst_findings,
            proposed_actions=proposed_actions,
            policies=policies,
        )


# Single shared instance — DSPy modules are stateless and thread-safe enough
# for the way we use them. Recreating per call would just add overhead.
_criticiser = _PolicyCriticiserModule()


# --- Public async API -----------------------------------------------------
async def critique(
    findings: str,
    plan: str,
    policies: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[str, List[CritiqueResult]]:
    """Run the policy criticiser against a plan.

    DSPy modules are synchronous; this wraps the call in `asyncio.to_thread`
    so the autogen event loop isn't blocked while the LLM responds.

    Returns `(reasoning, results)` where `results` contains one
    CritiqueResult per (step, policy) pair the model evaluated. The catalog
    defaults to the module-level POLICIES if none is passed.
    """
    catalog = policies if policies is not None else POLICIES
    prediction = await asyncio.to_thread(
        _criticiser.forward,
        analyst_findings=findings,
        proposed_actions=plan,
        policies=catalog,
    )
    return prediction.reasoning, list(prediction.results)
