"""
FILE: agents/statistician.py
------------------------------
Statistician agent — wires together all components and exposes
the public run_statistician() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt  ← prompts/statistician_prompt.py
  - Tools   ← tools/statistician_tools.py
  - Engine  ← core/statistician_engine.py  (called via tools)
  - Schemas ← schemas/statistician_schema.py

Note on fitted model passing:
  The fitted model (statsmodels OLS, sklearn Pipeline, etc.) is stored
  at module level in tools/statistician_tools.py via get_fitted_model().
  The orchestrator calls get_fitted_model() and passes it directly to
  run_model_critic() — no serialization needed.
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.statistician_prompt import STATISTICIAN_SYSTEM_PROMPT
from tools.statistician_tools import (
    STATISTICIAN_TOOLS,
    init_statistician_store,
    get_statistician_store,
    get_fitted_model,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_statistician_agent():
    """Create and return the Statistician ReAct agent."""
    return create_react_agent(
        model=model,
        tools=STATISTICIAN_TOOLS,
        prompt=STATISTICIAN_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_statistician(
    methodologist_output: dict,
    cleaned_df: pd.DataFrame,
    rectification_output: dict | None = None,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        methodologist_output: MethodologistOutput dict — selected test + column roles.
        cleaned_df:           Cleaned (and possibly rectified/transformed) DataFrame.
        rectification_output: RectificationOutput dict if rectification was applied.
                              Pass None if no assumption failures occurred.
                              correction_type and new_test are read automatically.

    Returns:
        {
          "messages":             Full LangGraph message history.
          "final_response":       Human-readable results shown to user.
          "statistician_output":  Raw StatisticianOutput dict for downstream agents.
          "fitted_model":         The fitted model object in memory.
                                  Pass this directly to run_model_critic().
                                  None for non-parametric and correlation tests.
        }
    """
    init_statistician_store(
        methodologist_output=methodologist_output,
        cleaned_df=cleaned_df,
        rectification_output=rectification_output,
    )

    agent = create_statistician_agent()
    content = "Please execute the statistical test and present the results."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve results from store ──
    store = get_statistician_store()
    statistician_output = store.get("statistician_output")
    statistician_output_dict = (
        statistician_output.model_dump() if statistician_output else {}
    )

    return {
        "messages":            result["messages"],
        "final_response":      final_response,
        "statistician_output": statistician_output_dict,
        "fitted_model":        get_fitted_model(),  # None for non-parametric tests
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/statistician.py
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     import numpy as np

#     np.random.seed(42)
#     mock_df = pd.DataFrame({
#         "salary": np.concatenate([
#             np.random.normal(55000, 12000, 110),
#             np.random.normal(50000, 11000, 90),
#         ]),
#         "gender": ["M"] * 110 + ["F"] * 90,
#     })

#     mock_methodologist_output = {
#         "selected_test":         "Independent Samples T-Test",
#         "selection_mode":        "decided",
#         "dependent_variable":    "salary",
#         "independent_variables": [],
#         "grouping_variable":     "gender",
#         "n_rows":                200,
#         "n_groups":              2,
#         "dependent_dtype":       "continuous",
#         "independent_dtypes":    [],
#         "mismatch_warning":      None,
#         "override_reason":       None,
#         "user_requested_test":   None,
#         "reasoning":             "T-Test selected for 2-group comparison.",
#         "original_query":        "Does salary differ between genders?",
#     }

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Statistician Agent")
#     print("=" * 60 + "\n")

#     output = run_statistician(
#         methodologist_output=mock_methodologist_output,
#         cleaned_df=mock_df,
#         rectification_output=None,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     print("Test name:    ", output["statistician_output"].get("test_name"))
#     print("Test family:  ", output["statistician_output"].get("test_family"))
#     print("Model in mem: ", output["fitted_model"] is not None)