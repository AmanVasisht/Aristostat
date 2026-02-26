"""
FILE: schemas/assumption_checker_schema.py
-------------------------------------------
Pydantic output schema for the Assumption Checker agent.
AssumptionCheckerOutput carries the full assumption check results
to the orchestrator, which then decides whether to proceed to the
Statistician or route to the Rectification Strategist.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class AssumptionStatus(str, Enum):
    PASSED  = "passed"   # assumption met
    FAILED  = "failed"   # assumption violated
    MANUAL  = "manual"   # cannot be verified programmatically — user must confirm
    WARNING = "warning"  # borderline — not a hard fail but worth noting


# ─────────────────────────────────────────────
# SINGLE ASSUMPTION RESULT
# ─────────────────────────────────────────────

class AssumptionResult(BaseModel):
    name: str                               # e.g. "normality", "homoscedasticity"
    description: str                        # plain English description
    status: AssumptionStatus

    # ── Statistical test details (populated for statistical_test checks) ──
    test_used: str | None = None            # e.g. "Shapiro-Wilk", "Levene's Test"
    statistic: float | None = None          # test statistic value
    p_value: float | None = None            # p-value
    alpha: float | None = None              # significance level used

    # ── Plain English explanation ──
    plain_reason: str = ""                  # e.g. "p=0.003 < 0.05, normality rejected"

    # ── For manual checks ──
    manual_confirmation_required: bool = False
    manual_question: str | None = None      # question shown to user for manual checks


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class AssumptionCheckerOutput(BaseModel):
    selected_test: str                      # test that was checked against

    # ── Individual assumption results ──
    results: list[AssumptionResult] = Field(default_factory=list)

    # ── Aggregated pass/fail counts ──
    total_assumptions: int = 0
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    manual_count: int = 0                   # assumptions requiring user confirmation

    # ── Overall verdict ──
    all_assumptions_met: bool = False       # True only if passed + manual == total
                                            # (manual assumed met pending user confirmation)
    has_failures: bool = False              # True if any FAILED results exist

    # ── Pending manual confirmations ──
    # List of assumption names that need user confirmation before proceeding.
    pending_manual_confirmations: list[str] = Field(default_factory=list)

    # ── For orchestrator routing ──
    # If True  → proceed to Statistician
    # If False → route to Rectification Strategist (or ask user)
    proceed_to_statistician: bool = False

    # ── Summary shown to user ──
    summary_message: str = ""