"""
FILE: agents/preprocessor.py
------------------------------
Preprocessor agent — wires together all components and exposes
the public run_preprocessor() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/preprocessor_prompt.py
  - Tools   ← tools/preprocessor_tools.py
  - Engine  ← core/preprocessor_engine.py  (called via tools)
  - Schemas ← schemas/preprocessor_schema.py

Note on state passing:
  The cleaned DataFrame is stored in two ways:
    1. As a pd.DataFrame in memory (cleaned_df) — used by the next agent directly
       if running in the same process (recommended for performance).
    2. As a JSON string in PreprocessorOutput.cleaned_data_json — travels in
       LangGraph state and can be reconstructed with pd.read_json() anywhere.
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.preprocessor_engine import preprocess_dataframe


# ─────────────────────────────────────────────
# LLM — used only for formatting the summary
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_preprocessor(
    raw_df: pd.DataFrame,
    profiler_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    Calls the preprocessor engine directly — no LLM needed for cleaning.
    The LLM is only used to format the final human-readable summary.
    """
    cleaned_df, preprocessor_output = preprocess_dataframe(
        df=raw_df,
        profiler_output=profiler_output,
    )

    preprocessor_output_dict = preprocessor_output.model_dump()

    # ── Format human-readable summary via LLM ──
    changes = preprocessor_output.changes_summary
    warnings = preprocessor_output.warnings
    fatal = preprocessor_output.fatal_error

    if fatal:
        final_response = f"⚠️ Fatal error during preprocessing: {fatal}"
    elif not changes:
        final_response = "Your data required no cleaning — it is already in good shape."
    else:
        summary_prompt = f"""Summarize these data cleaning steps in 3-5 plain English sentences for a non-technical user.
Be concise. List what was done and the final dataset shape.

Changes made: {changes}
Warnings: {warnings}
Original shape: {preprocessor_output.original_shape}
Final shape: {preprocessor_output.final_shape}
Rows dropped: {preprocessor_output.rows_dropped_total}"""

        response = model.invoke([HumanMessage(content=summary_prompt)])
        final_response = response.content.strip()

    return {
        "messages":            [],
        "final_response":      final_response,
        "preprocessor_output": preprocessor_output_dict,
        "cleaned_df":          cleaned_df,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/preprocessor.py <path_to_csv>
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python agents/preprocessor.py <path_to_csv>")
#         sys.exit(1)

#     import json

#     csv_path = sys.argv[1]
#     raw_df = pd.read_csv(csv_path)

#     # ── Minimal mock profiler output for standalone testing ──
#     # In real usage this comes from run_data_profiler()
#     mock_profiler = {
#         "n_rows": len(raw_df),
#         "n_cols": len(raw_df.columns),
#         "continuous_columns": [
#             {
#                 "column": col,
#                 "missing_pct": float(raw_df[col].isna().mean()),
#                 "missing_severity": (
#                     "low" if raw_df[col].isna().mean() < 0.05
#                     else "moderate" if raw_df[col].isna().mean() < 0.20
#                     else "high"
#                 ),
#                 "disguised_nulls_found": [],
#                 "disguised_null_count": 0,
#             }
#             for col in raw_df.select_dtypes(include="number").columns
#         ],
#         "categorical_columns": [
#             {
#                 "column": col,
#                 "missing_pct": float(raw_df[col].isna().mean()),
#                 "missing_severity": (
#                     "low" if raw_df[col].isna().mean() < 0.05
#                     else "moderate" if raw_df[col].isna().mean() < 0.20
#                     else "high"
#                 ),
#                 "disguised_nulls_found": [],
#                 "disguised_null_count": 0,
#             }
#             for col in raw_df.select_dtypes(include="object").columns
#         ],
#         "warnings": [],
#         "fatal_errors": [],
#     }

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Preprocessor Agent")
#     print("=" * 60 + "\n")

#     output = run_preprocessor(raw_df=raw_df, profiler_output=mock_profiler)

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     if output["cleaned_df"] is not None:
#         print(f"Cleaned shape: {output['cleaned_df'].shape}")
#     print("Fatal error:", output["preprocessor_output"].get("fatal_error"))