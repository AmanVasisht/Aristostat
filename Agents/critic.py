"""
FILE: agents/model_critic.py
------------------------------
Model Critic agent — direct engine call, no ReAct agent.
LLM used only for formatting the final response.

Imports:
  - Engine   ← core/model_critic_engine.py (called directly)
  - Schemas  ← schemas/model_critic_schema.py
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.critic_engine import run_post_test_checks


# ─────────────────────────────────────────────
# LLM — used only for formatting the response
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
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
    Calls model_critic_engine directly — no ReAct agent.
    LLM used only to format the human-readable response.

    Args:
        statistician_output:  StatisticianOutput dict — test results + test_family.
        fitted_model:         Fitted model object from Statistician (in memory).
                              Passed directly — not serialized.
                              None for inference, correlation, dimensionality tests.
        cleaned_df:           Current working DataFrame.
        methodologist_output: MethodologistOutput dict — column roles.

    Returns:
        {
          "messages":          Empty list (no agent messages).
          "final_response":    Human-readable post-test check summary shown to user.
          "critic_output":     Raw ModelCriticOutput dict for orchestrator routing.
                               Key fields:
                                 - checks_applicable:       False if non-regression test
                                 - has_failures:            True if any check failed
                                 - proceed_to_final_report: True if safe to proceed
        }
    """
    critic_output = run_post_test_checks(
        statistician_output=statistician_output,
        fitted_model=fitted_model,
        cleaned_df=cleaned_df,
        methodologist_output=methodologist_output,
    )

    critic_output_dict = critic_output.model_dump()

    # ── Format response via LLM ──
    if not critic_output.checks_applicable:
        final_response = (
            f"Post-test model checks are not applicable for "
            f"{statistician_output.get('test_name', 'this test')}. "
            f"Proceeding to final report."
        )
    else:
        format_prompt = f"""Format these post-test model check results clearly for the user.
Use ✅ for passed, ❌ for failed, ⚠️ for warnings.
For each check show: name, method used, key statistic, and plain English verdict.
End with a clear summary of how many passed/failed and what action is recommended.

Results:
{critic_output.summary_message}

Has failures: {critic_output.has_failures}
Failed count: {critic_output.failed_count}
Warning count: {critic_output.warning_count}"""

        response = model.invoke([HumanMessage(content=format_prompt)])
        final_response = response.content.strip()

    return {
        "messages":       [],
        "final_response": final_response,
        "critic_output":  critic_output_dict,
    }