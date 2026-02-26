"""
FILE: agents/model_critic.py
------------------------------
Model Critic agent — wires together all components and exposes
the public run_model_critic() entry point for the LangGraph orchestrator.

Imports:
  - LLM model
  - Prompt   ← prompts/model_critic_prompt.py
  - Tools    ← tools/model_critic_tools.py
  - Engine   ← core/model_critic_engine.py  (called via tools)
  - Config   ← configs/post_test_assumptions.py
  - Schemas  ← schemas/model_critic_schema.py
"""

from typing import Any

import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.model_critic_prompt import MODEL_CRITIC_SYSTEM_PROMPT
from tools.model_critic_tools import (
    MODEL_CRITIC_TOOLS,
    init_model_critic_store,
    get_model_critic_store,
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

def create_model_critic_agent():
    """Create and return the Model Critic ReAct agent."""
    return create_react_agent(
        model=model,
        tools=MODEL_CRITIC_TOOLS,
        prompt=MODEL_CRITIC_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_model_critic(
    statistician_output: dict,
    fitted_model: object | None,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).

    Args:
        statistician_output:  StatisticianOutput dict — test results + test_family.
        fitted_model:         Fitted model object from Statistician (in memory).
                              Passed directly — not serialized.
                              None for inference, correlation, dimensionality tests.
        cleaned_df:           Current working DataFrame.
        methodologist_output: MethodologistOutput dict — column roles.

    Returns:
        {
          "messages":          Full LangGraph message history.
          "final_response":    Human-readable post-test check summary shown to user.
          "critic_output":     Raw ModelCriticOutput dict for orchestrator routing.
                               Key fields:
                                 - checks_applicable:       False if non-regression test
                                 - has_failures:            True if any check failed
                                 - proceed_to_final_report: True if safe to proceed
        }
    """
    init_model_critic_store(
        statistician_output=statistician_output,
        fitted_model=fitted_model,
        cleaned_df=cleaned_df,
        methodologist_output=methodologist_output,
    )

    agent = create_model_critic_agent()
    content = "Please run post-test model checks and present the results to the user."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve critic output from store ──
    store = get_model_critic_store()
    critic_output = store.get("critic_output")
    critic_output_dict = critic_output.model_dump() if critic_output else {}

    return {
        "messages":       result["messages"],
        "final_response": final_response,
        "critic_output":  critic_output_dict,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/model_critic.py
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     from statsmodels.regression.linear_model import OLS
#     from statsmodels.tools import add_constant

#     # ── Mock regression data ──
#     np.random.seed(42)
#     n = 150
#     experience = np.random.uniform(1, 20, n)
#     salary = 30000 + 2000 * experience + np.random.normal(0, 5000, n)

#     mock_df = pd.DataFrame({"salary": salary, "experience": experience})

#     # ── Fit a real OLS model for testing ──
#     X = add_constant(mock_df[["experience"]])
#     fitted = OLS(mock_df["salary"], X).fit()

#     mock_statistician_output = {
#         "test_name":   "Simple Linear Regression",
#         "test_family": "regression",
#         "n_observations": n,
#         "correction_applied": None,
#         "model_available_in_memory": True,
#         "columns_used": {"dependent": "salary", "independent": "experience"},
#     }

#     mock_methodologist_output = {
#         "selected_test":         "Simple Linear Regression",
#         "selection_mode":        "decided",
#         "dependent_variable":    "salary",
#         "independent_variables": ["experience"],
#         "grouping_variable":     None,
#         "n_rows":                n,
#         "reasoning":             "SLR selected.",
#         "original_query":        "Does experience predict salary?",
#     }

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Model Critic Agent")
#     print("=" * 60 + "\n")

#     output = run_model_critic(
#         statistician_output=mock_statistician_output,
#         fitted_model=fitted,
#         cleaned_df=mock_df,
#         methodologist_output=mock_methodologist_output,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     print("Has failures:          ", output["critic_output"].get("has_failures"))
#     print("Proceed to final report:", output["critic_output"].get("proceed_to_final_report"))
#     print("Checks applicable:     ", output["critic_output"].get("checks_applicable"))