"""
FILE: schemas/methodologist_schema.py
---------------------------------------
Pydantic output schema for the Methodologist agent.
MethodologistOutput is the structured contract passed to the
Assumption Checker and all downstream agents.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class SelectionMode(str, Enum):
    BYPASS      = "bypass"      # user named test, column types are compatible — passed through
    DECIDED     = "decided"     # methodologist picked the test via decision tree
    WARNED      = "warned"      # user named test, minor mismatch found — warned but proceeded
    OVERRIDDEN  = "overridden"  # user named test but it was clearly wrong — overridden with correct test


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class MethodologistOutput(BaseModel):
    # ── Selected test ──
    selected_test: str                          # canonical test name e.g. "One-Way ANOVA"
    selection_mode: SelectionMode               # how the test was selected

    # ── Only populated when selection_mode is OVERRIDDEN ──
    user_requested_test: str | None = None      # what the user originally asked for
    override_reason: str | None = None          # plain English explanation of why it was overridden

    # ── Only populated when selection_mode is WARNED ──
    mismatch_warning: str | None = None         # shown to user — minor issue, still using their test

    # ── Columns confirmed for this test ──
    dependent_variable: str | None = None
    independent_variables: list[str] = Field(default_factory=list)
    grouping_variable: str | None = None

    # ── Decision factors used (for audit trail) ──
    n_rows: int | None = None
    n_groups: int | None = None
    dependent_dtype: str | None = None
    independent_dtypes: list[str] = Field(default_factory=list)

    # ── Reasoning shown to user (always populated) ──
    reasoning: str = ""

    # ── Raw input preserved for downstream ──
    original_query: str = ""