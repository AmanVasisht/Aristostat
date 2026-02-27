"""
FILE: schemas/model_critic_schema.py
--------------------------------------
Pydantic output schema for the Model Critic agent.
Reuses AssumptionResult and AssumptionStatus from the
Assumption Checker schema — same structure, different checks.
"""

from pydantic import BaseModel, Field
from Schemas.assumption_checker import AssumptionResult, AssumptionStatus


class ModelCriticOutput(BaseModel):
    test_name:   str
    test_family: str                    # "regression" | "inference" | "correlation" | "dimensionality"

    # ── Whether post-test checks were run ──
    checks_applicable: bool = True      # False for inference/correlation/dimensionality

    # ── Individual check results ──
    results: list[AssumptionResult] = Field(default_factory=list)

    # ── Aggregated counts ──
    total_checks:  int = 0
    passed_count:  int = 0
    failed_count:  int = 0
    warning_count: int = 0

    # ── Overall verdict ──
    has_failures: bool = False

    # ── Orchestrator routing ──
    # True  → proceed to Final Report
    # False → route to Rectification Strategist (post_test phase)
    proceed_to_final_report: bool = True

    # ── Summary shown to user ──
    summary_message: str = ""