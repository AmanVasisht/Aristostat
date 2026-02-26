"""
FILE: schemas/preprocessor_schema.py
--------------------------------------
Pydantic output schema for the Preprocessor agent.
PreprocessorOutput carries the cleaning summary and the
cleaned DataFrame (serialized) forward in LangGraph state.
"""

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# PER-COLUMN ACTION LOG
# Records exactly what was done to each column.
# ─────────────────────────────────────────────

class ColumnCleaningLog(BaseModel):
    column: str
    disguised_nulls_replaced: int = 0      # count of disguised null symbols replaced with NaN
    nulls_imputed: int = 0                  # count of NaN values filled via imputation
    rows_dropped_due_to_null: int = 0       # rows dropped because of NaN in this column
    dtype_coerced: bool = False             # True if object dtype was coerced to numeric
    original_dtype: str | None = None
    final_dtype: str | None = None
    imputation_strategy: str | None = None  # "mean" | "median" | "mode" | None


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class PreprocessorOutput(BaseModel):
    # ── Shape changes ──
    original_shape: tuple[int, int]
    final_shape: tuple[int, int]
    rows_dropped_total: int = 0

    # ── Per-column log ──
    column_logs: list[ColumnCleaningLog] = Field(default_factory=list)

    # ── High missingness columns that were flagged as errors ──
    # These columns had >20% missing and could not be safely handled.
    # Pipeline stops and user must decide what to do.
    high_missingness_fatal: list[str] = Field(default_factory=list)
    fatal_error: str | None = None          # set if high_missingness_fatal is non-empty

    # ── Summary for user display ──
    changes_summary: list[str] = Field(default_factory=list)   # plain English change log
    warnings: list[str] = Field(default_factory=list)          # non-fatal concerns

    # ── Cleaned data — stored as JSON string for LangGraph state ──
    # Downstream agents reconstruct DataFrame with pd.read_json(cleaned_data_json)
    cleaned_data_json: str = ""