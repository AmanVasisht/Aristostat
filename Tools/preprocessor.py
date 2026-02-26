"""
FILE: tools/preprocessor_tools.py
------------------------------------
LangChain tools exposed to the Preprocessor ReAct agent.
The agent uses these to access the raw DataFrame and profiler output,
run the preprocessing engine, and store the cleaned result.
"""

import json
import pandas as pd
from langchain_core.tools import tool

from core.preprocessor_engine import preprocess_dataframe


# ─────────────────────────────────────────────
# SESSION STORE
# Holds the raw DataFrame, profiler output, and
# the cleaned DataFrame + PreprocessorOutput after processing.
# ─────────────────────────────────────────────

_preprocessor_store: dict = {
    "raw_df":               None,   # pd.DataFrame — set before agent invoked
    "profiler_output":      None,   # ProfilerOutput dict — set before agent invoked
    "cleaned_df":           None,   # pd.DataFrame — set after run_preprocessing called
    "preprocessor_output":  None,   # PreprocessorOutput — set after run_preprocessing called
}


def init_preprocessor_store(raw_df: pd.DataFrame, profiler_output: dict) -> None:
    """
    Called by run_preprocessor() before invoking the agent.
    Seeds the store with the raw DataFrame and profiler output.
    """
    _preprocessor_store["raw_df"]              = raw_df
    _preprocessor_store["profiler_output"]     = profiler_output
    _preprocessor_store["cleaned_df"]          = None
    _preprocessor_store["preprocessor_output"] = None


def get_preprocessor_store() -> dict:
    """Expose store to the agent runner for retrieving cleaned DataFrame and output."""
    return _preprocessor_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_data_issues_summary() -> str:
    """
    Returns a summary of data quality issues identified by the Data Profiler
    that need to be addressed during preprocessing.
    Includes: disguised nulls per column, missing value severity per column,
    and columns flagged for dtype coercion.
    Call this first to understand what needs to be fixed.
    """
    profiler = _preprocessor_store.get("profiler_output")
    if not profiler:
        return "ERROR: No profiler output in store."

    issues: dict = {
        "n_rows_before_cleaning": profiler.get("n_rows"),
        "columns_with_disguised_nulls": [],
        "columns_with_missing_values": [],
        "columns_needing_dtype_coercion": [],
    }

    all_cols = (
        profiler.get("continuous_columns", []) +
        profiler.get("categorical_columns", [])
    )

    for col in all_cols:
        name = col["column"]

        if col.get("disguised_nulls_found"):
            issues["columns_with_disguised_nulls"].append({
                "column":  name,
                "symbols": col["disguised_nulls_found"],
                "count":   col.get("disguised_null_count", 0),
            })

        if col.get("missing_pct", 0) > 0:
            issues["columns_with_missing_values"].append({
                "column":   name,
                "missing_pct": col.get("missing_pct"),
                "severity": col.get("missing_severity"),
            })

    # Columns listed as continuous but originally object dtype need coercion
    raw_df = _preprocessor_store.get("raw_df")
    if raw_df is not None:
        for col in profiler.get("continuous_columns", []):
            name = col["column"]
            if name in raw_df.columns and raw_df[name].dtype == object:
                issues["columns_needing_dtype_coercion"].append(name)

    return json.dumps(issues, indent=2)


@tool
def run_preprocessing() -> str:
    """
    Runs the full preprocessing pipeline on the raw DataFrame.
    Performs three steps in order:
      1. Replace disguised null symbols with proper NaN
      2. Coerce object-dtype columns to numeric where appropriate
      3. Handle missing values:
           - < 5% missing  → drop affected rows
           - 5-20% missing → impute (median for continuous, mode for categorical)
           - > 20% missing → FATAL ERROR — pipeline cannot continue

    Returns the PreprocessorOutput as a JSON string.
    If fatal_error is set in the output, inform the user and stop.
    """
    raw_df  = _preprocessor_store.get("raw_df")
    profiler = _preprocessor_store.get("profiler_output")

    if raw_df is None:
        return "ERROR: No raw DataFrame in store."
    if profiler is None:
        return "ERROR: No profiler output in store."

    try:
        cleaned_df, preprocessor_output = preprocess_dataframe(raw_df, profiler)
        _preprocessor_store["cleaned_df"]          = cleaned_df
        _preprocessor_store["preprocessor_output"] = preprocessor_output
        return preprocessor_output.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Preprocessing failed — {str(e)}"


@tool
def get_cleaned_data_sample(n: int = 5) -> str:
    """
    Returns the first n rows of the cleaned DataFrame as a JSON string.
    Use this after run_preprocessing to verify the data looks correct
    before presenting the summary to the user.
    Args:
        n: Number of rows to return (default 5, max 20).
    """
    cleaned_df = _preprocessor_store.get("cleaned_df")
    if cleaned_df is None:
        return "ERROR: No cleaned DataFrame available. Run run_preprocessing first."
    n = min(n, 20)
    return cleaned_df.head(n).to_json(orient="records", indent=2)


@tool
def get_shape_comparison() -> str:
    """
    Returns a before/after shape comparison of the DataFrame.
    Useful for confirming how many rows were dropped during preprocessing.
    """
    output = _preprocessor_store.get("preprocessor_output")
    if output is None:
        return "ERROR: No preprocessor output available. Run run_preprocessing first."

    return json.dumps({
        "original_shape": list(output.original_shape),
        "final_shape":    list(output.final_shape),
        "rows_dropped":   output.rows_dropped_total,
    }, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

PREPROCESSOR_TOOLS = [
    get_data_issues_summary,
    run_preprocessing,
    get_cleaned_data_sample,
    get_shape_comparison,
]