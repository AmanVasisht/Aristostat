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
from langgraph.prebuilt import create_react_agent

from Prompts.preprocessor import PREPROCESSOR_SYSTEM_PROMPT
from Tools.preprocessor import (
    PREPROCESSOR_TOOLS,
    init_preprocessor_store,
    get_preprocessor_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

from langchain_groq import ChatGroq
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_preprocessor_agent():
    """
    Create and return the Preprocessor ReAct agent.
    Called fresh per invocation to avoid stale state.
    """
    return create_react_agent(
        model=model,
        tools=PREPROCESSOR_TOOLS,
        prompt=PREPROCESSOR_SYSTEM_PROMPT
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

    Args:
        raw_df:          The original raw DataFrame loaded from the user's CSV.
        profiler_output: The ProfilerOutput dict from the Data Profiler agent.

    Returns:
        {
          "messages":             Full LangGraph message history.
          "final_response":       Human-readable cleaning summary shown to user.
          "preprocessor_output":  Raw PreprocessorOutput dict for LangGraph state.
                                  Includes cleaned_data_json for state serialization.
          "cleaned_df":           The actual cleaned pd.DataFrame for direct use
                                  by the next agent in the same process.
        }
    """
    init_preprocessor_store(
        raw_df=raw_df,
        profiler_output=profiler_output,
    )

    agent = create_preprocessor_agent()

    content = "Please clean and preprocess the dataset so it is ready for analysis."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve results from store ──
    store = get_preprocessor_store()
    preprocessor_output = store.get("preprocessor_output")
    cleaned_df = store.get("cleaned_df")

    preprocessor_output_dict = (
        preprocessor_output.model_dump() if preprocessor_output else {}
    )

    return {
        "messages":            result["messages"],
        "final_response":      final_response,
        "preprocessor_output": preprocessor_output_dict,
        "cleaned_df":          cleaned_df,      # pd.DataFrame — pass directly to next agent
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