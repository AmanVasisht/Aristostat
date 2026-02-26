"""
FILE: agents/rectification_strategist.py
------------------------------------------
Rectification Strategist agent — wires together all components and exposes
the public run_rectification_strategist() entry point for the LangGraph orchestrator.

Reused for both pre-test failures (called from Assumption Checker route)
and post-test failures (called from Model Critic route).
The phase flag distinguishes the two contexts.

Imports:
  - LLM model
  - Prompt   ← prompts/rectification_prompt.py
  - Tools    ← tools/rectification_tools.py
  - Engine   ← core/rectification_engine.py  (called via tools)
  - Config   ← configs/rectifications.py
  - Schemas  ← schemas/rectification_schema.py
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.rectification_prompt import RECTIFICATION_STRATEGIST_SYSTEM_PROMPT
from tools.rectification_tools import (
    RECTIFICATION_TOOLS,
    init_rectification_store,
    get_rectification_store,
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

def create_rectification_agent():
    """Create and return the Rectification Strategist ReAct agent."""
    return create_react_agent(
        model=model,
        tools=RECTIFICATION_TOOLS,
        prompt=RECTIFICATION_STRATEGIST_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_rectification_strategist(
    failed_assumptions: list[str],
    phase: str,                         # "pre_test" | "post_test"
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
    rectification_attempt: int = 1,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    Called when Assumption Checker OR Model Critic finds failures.

    Args:
        failed_assumptions:   List of assumption names that failed.
        phase:                "pre_test" (from Assumption Checker) or
                              "post_test" (from Model Critic).
        cleaned_df:           Current working DataFrame (may have been
                              transformed in a previous rectification attempt).
        methodologist_output: MethodologistOutput dict — used for column roles.
        rectification_attempt: Which attempt this is (incremented by orchestrator).
        max_attempts:         Maximum rectification loops allowed (default 3).

    Returns:
        {
          "messages":              Full LangGraph message history.
          "final_response":        Human-readable summary shown to user.
          "rectification_output":  Raw RectificationOutput dict.
                                   Key fields:
                                     - next_step: "assumption_checker" | "statistician"
                                     - new_test:  set if test was switched
                                     - correction_type: set if correction applied
                                     - applied_transforms: list of data changes
                                     - user_accepted_violation: bool
          "rectified_df":          Updated pd.DataFrame after applying solution.
                                   Pass this to the next agent instead of original df.
        }
    """
    init_rectification_store(
        failed_assumptions=failed_assumptions,
        phase=phase,
        cleaned_df=cleaned_df,
        methodologist_output=methodologist_output,
        rectification_attempt=rectification_attempt,
        max_attempts=max_attempts,
    )

    agent = create_rectification_agent()

    content = (
        f"Assumption failures detected during {'pre-test' if phase == 'pre_test' else 'post-test'} "
        f"checking. Please propose rectification solutions to the user."
    )

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve results from store ──
    store = get_rectification_store()
    rectification_output = store.get("rectification_output")
    rectified_df = store.get("cleaned_df")          # may be transformed

    rectification_output_dict = (
        rectification_output.model_dump() if rectification_output else {}
    )

    return {
        "messages":             result["messages"],
        "final_response":       final_response,
        "rectification_output": rectification_output_dict,
        "rectified_df":         rectified_df,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/rectification_strategist.py
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     import numpy as np

#     # ── Mock: normality failed for an Independent T-Test ──
#     np.random.seed(42)
#     mock_df = pd.DataFrame({
#         "salary": np.random.exponential(scale=50000, size=200),  # skewed — normality will fail
#         "gender": np.random.choice(["M", "F"], 200),
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
#     print("  ARISTOSTAT — Rectification Strategist Agent")
#     print("=" * 60 + "\n")

#     output = run_rectification_strategist(
#         failed_assumptions=["normality", "homogeneity_of_variance"],
#         phase="pre_test",
#         cleaned_df=mock_df,
#         methodologist_output=mock_methodologist_output,
#         rectification_attempt=1,
#         max_attempts=3,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     rect_out = output["rectification_output"]
#     print("Next step:         ", rect_out.get("next_step"))
#     print("New test:          ", rect_out.get("new_test"))
#     print("Correction type:   ", rect_out.get("correction_type"))
#     print("Accepted violation:", rect_out.get("user_accepted_violation"))
#     print("Applied transforms:", rect_out.get("applied_transforms"))