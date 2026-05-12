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
# Verbatim policy statements. The DSPy module interprets each one against
# the findings + plan. Order matters: results come back in the same order.
POLICIES: List[str] = [
    "make sure that the owners are the ones that have the authority",
    "upstream dependencies are done",
]


# --- Structured verdict shape ---------------------------------------------
class CritiqueResult(BaseModel):
    """One verdict per policy. Rendered as a passed/failed line in the LogicAgent."""

    policy: str
    passed: bool
    reason: str


# --- DSPy signature -------------------------------------------------------
class PolicyCritique(dspy.Signature):
    """Audit a proposed action plan against a list of policies.

    For each policy in the input list, decide whether the plan satisfies it,
    using the analyst findings as your only source of facts. Return one
    CritiqueResult per policy, in the same order as the input list:
      - policy: echo the policy statement verbatim
      - passed: true if the plan clearly satisfies the policy, false otherwise
      - reason: one or two sentences explaining the verdict, citing task_ids,
        evidence_ids, decision_ids, owners, etc. verbatim from the findings

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
    policies: list[str] = dspy.InputField(
        desc="Policy catalog to evaluate against. Process in order.",
    )
    results: list[CritiqueResult] = dspy.OutputField(
        desc="One verdict per policy, in the same order as the input list.",
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
        policies: list[str],
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
    policies: List[str] | None = None,
) -> List[CritiqueResult]:
    """Run the policy criticiser against a plan.

    DSPy modules are synchronous; this wraps the call in `asyncio.to_thread`
    so the autogen event loop isn't blocked while the LLM responds.

    Returns one CritiqueResult per policy, in the same order as the input
    (defaulting to the module-level POLICIES catalog).
    """
    catalog = policies if policies is not None else POLICIES
    prediction = await asyncio.to_thread(
        _criticiser.forward,
        analyst_findings=findings,
        proposed_actions=plan,
        policies=catalog,
    )
    return list(prediction.results)
