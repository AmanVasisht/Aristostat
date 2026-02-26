"""
FILE: schemas/rectification_schema.py
---------------------------------------
Pydantic output schema for the Rectification Strategist agent.
RectificationOutput carries the chosen solution and routing
instruction back to the orchestrator.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class RectificationPhase(str, Enum):
    PRE_TEST  = "pre_test"   # called from Assumption Checker failures
    POST_TEST = "post_test"  # called from Model Critic failures


class NextStep(str, Enum):
    ASSUMPTION_CHECKER = "assumption_checker"  # re-run assumption checks
    STATISTICIAN       = "statistician"        # proceed directly to test
    FINISH             = "finish"              # user chose to proceed despite failure


# ─────────────────────────────────────────────
# PROPOSED SOLUTION
# ─────────────────────────────────────────────

class ProposedSolution(BaseModel):
    solution_id:   str
    description:   str
    action_type:   str          # "transform" | "test_switch" | "correction" | "drop"
    action_details: dict = Field(default_factory=dict)
    next_step:     str          # where to route after applying this solution


# ─────────────────────────────────────────────
# APPLIED TRANSFORM LOG
# Tracks what data changes were made if a transform solution was chosen.
# ─────────────────────────────────────────────

class AppliedTransform(BaseModel):
    column: str
    transform_type: str         # "log_transform" | "sqrt_transform" | "first_difference"
    original_min: float | None = None
    original_max: float | None = None
    note: str = ""              # e.g. "log1p used to handle zero values"


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class RectificationOutput(BaseModel):
    # ── Context ──
    phase: RectificationPhase
    failed_assumptions: list[str] = Field(default_factory=list)

    # ── Proposed solutions shown to user ──
    proposed_solutions: list[ProposedSolution] = Field(default_factory=list)

    # ── Chosen solution (populated after user picks) ──
    chosen_solution_id: str | None = None
    chosen_solution: ProposedSolution | None = None

    # ── Routing ──
    next_step: NextStep = NextStep.ASSUMPTION_CHECKER

    # ── If solution involved a test switch ──
    new_test: str | None = None             # canonical name of the new test

    # ── If solution involved a data transform ──
    applied_transforms: list[AppliedTransform] = Field(default_factory=list)
    rectified_data_json: str = ""           # serialized DataFrame after transform

    # ── If solution involved a correction ──
    correction_type: str | None = None      # e.g. "welch", "robust_se", "newey_west"

    # ── Iteration tracking (prevents infinite loops) ──
    rectification_attempt: int = 1          # incremented each time this agent is called
    max_attempts: int = 3                   # after this, force user to proceed or stop

    # ── User chose to proceed despite failure ──
    user_accepted_violation: bool = False
    accepted_violation_names: list[str] = Field(default_factory=list)