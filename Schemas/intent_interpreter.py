"""
FILE: schemas/intent_schema.py
--------------------------------
Pydantic output schema for the Intent Interpreter agent.
IntentOutput is the structured contract passed to the Methodologist
and all downstream agents.

Behaviour encoded in schema:
  - clarification_needed is always False — ambiguity handled via low confidence flag.
  - suggested_combinations populated only for open-ended queries.
  - invalid_columns should always be empty on a successful run (hard error raised before).
"""

from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class IntentType(str, Enum):
    EXPLICIT_TEST       = "explicit_test"       # user named a specific test
    COLUMN_RELATIONSHIP = "column_relationship" # user described columns to analyse
    OPEN_ENDED          = "open_ended"          # layperson, no specific direction


class AnalysisGoal(str, Enum):
    PREDICTION          = "prediction"          # e.g. regression, ridge
    INFERENCE           = "inference"           # e.g. t-test, ANOVA, Mann-Whitney
    DIMENSIONALITY      = "dimensionality"      # e.g. PCA
    RELATIONSHIP        = "relationship"        # e.g. correlation, SLR
    DISTRIBUTION        = "distribution"        # e.g. normality check, descriptive only
    UNKNOWN             = "unknown"             # open-ended, let Methodologist decide


class ConfidenceLevel(str, Enum):
    HIGH    = "high"    # intent is clear and unambiguous
    MEDIUM  = "medium"  # reasonable guess, minor ambiguity
    LOW     = "low"     # significant ambiguity — best guess made, Methodologist should verify


# ─────────────────────────────────────────────
# COLUMN REFERENCE
# ─────────────────────────────────────────────

class ColumnReference(BaseModel):
    name: str
    role: str | None = None     # "dependent" | "independent" | "grouping" | "unspecified"
    dtype_from_profiler: str | None = None  # "continuous" | "categorical"


# ─────────────────────────────────────────────
# COLUMN COMBINATION SUGGESTION
# Populated only for open-ended queries.
# Each entry suggests a meaningful column pairing
# along with a recommended analysis direction.
# ─────────────────────────────────────────────

class ColumnCombination(BaseModel):
    columns: list[str]              # column names involved in this combination
    suggested_goal: str             # "relationship" | "inference" | "prediction" | "dimensionality"
    rationale: str                  # plain English explanation shown to the user


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class IntentOutput(BaseModel):
    # ── Core intent classification ──
    intent_type: IntentType
    analysis_goal: AnalysisGoal
    confidence: ConfidenceLevel     # LOW = ambiguous but best guess made, not a blocker

    # ── Test info (populated if intent_type is EXPLICIT_TEST) ──
    requested_test: str | None = None
    methodologist_bypass: bool = False  # True if user explicitly named a test

    # ── Columns involved ──
    columns: list[ColumnReference] = Field(default_factory=list)
    all_columns_mode: bool = False      # True for open-ended queries

    # ── Open-ended suggestions (populated only when intent_type is OPEN_ENDED) ──
    suggested_combinations: list[ColumnCombination] = Field(default_factory=list)

    # ── Validation results ──
    # invalid_columns should always be empty on successful run.
    # Any invalid column raises a hard error before IntentOutput is built.
    invalid_columns: list[str] = Field(default_factory=list)
    column_warnings: list[str] = Field(default_factory=list)    # e.g. case-insensitive matches

    # ── User-facing explanation ──
    interpretation_summary: str = ""

    # Clarification is never requested — ambiguity handled via low confidence.
    # Fields kept for schema completeness but always False/None.
    clarification_needed: bool = False
    clarification_question: str | None = None

    # ── Raw user query (preserved for audit trail) ──
    original_query: str = ""