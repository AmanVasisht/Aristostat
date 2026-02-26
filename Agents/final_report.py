"""
FILE: agents/final_report.py
------------------------------
Final Report agent — wires together all components and exposes
the public run_final_report() entry point for the LangGraph orchestrator.

This is the last agent in the pipeline. It receives all upstream outputs,
assembles a structured report, renders it in chat as markdown, and
generates a downloadable .docx Word document.

Imports:
  - LLM model
  - Prompt   ← prompts/final_report_prompt.py
  - Tools    ← tools/final_report_tools.py
  - Engine   ← core/final_report_engine.py  (called via tools)
  - Schemas  ← schemas/final_report_schema.py
"""

from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.final_report_prompt import FINAL_REPORT_SYSTEM_PROMPT
from tools.final_report_tools import (
    FINAL_REPORT_TOOLS,
    init_report_store,
    get_report_store,
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ─────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────

def create_final_report_agent():
    """Create and return the Final Report ReAct agent."""
    return create_react_agent(
        model=model,
        tools=FINAL_REPORT_TOOLS,
        prompt=FINAL_REPORT_SYSTEM_PROMPT,
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def run_final_report(
    original_query: str,
    profiler_output: dict,
    preprocessor_output: dict,
    methodologist_output: dict,
    checker_output: dict,
    statistician_output: dict,
    rectification_output: dict | None = None,
    critic_output: dict | None = None,
) -> dict[str, Any]:
    """
    Entry point called by the LangGraph orchestrator (main.py).
    This is the terminal node — nothing follows it.

    Args:
        original_query:       The user's original question/request.
        profiler_output:      ProfilerOutput dict.
        preprocessor_output:  PreprocessorOutput dict.
        methodologist_output: MethodologistOutput dict.
        checker_output:       AssumptionCheckerOutput dict.
        statistician_output:  StatisticianOutput dict.
        rectification_output: RectificationOutput dict (None if no failures).
        critic_output:        ModelCriticOutput dict (None if non-regression test).

    Returns:
        {
          "messages":       Full LangGraph message history.
          "final_response": The complete markdown report shown in chat.
          "report_output":  Raw FinalReportOutput dict.
                            Key fields:
                              - markdown_report: full markdown string
                              - docx_path:       path to generated .docx file
                              - docx_generated:  bool
                              - caveats:         list of caveats
        }
    """
    init_report_store(
        original_query=original_query,
        profiler_output=profiler_output,
        preprocessor_output=preprocessor_output,
        methodologist_output=methodologist_output,
        checker_output=checker_output,
        statistician_output=statistician_output,
        rectification_output=rectification_output,
        critic_output=critic_output,
    )

    agent = create_final_report_agent()
    content = "Please generate the final analysis report."

    result = agent.invoke({"messages": [HumanMessage(content=content)]})

    # ── Extract final human-readable response ──
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            final_response = msg.content
            break

    # ── Retrieve report output from store ──
    store = get_report_store()
    report_output = store.get("report_output")
    report_output_dict = report_output.model_dump() if report_output else {}

    return {
        "messages":       result["messages"],
        "final_response": final_response,
        "report_output":  report_output_dict,
    }


# ─────────────────────────────────────────────
# STANDALONE TEST RUNNER
# Usage: python agents/final_report.py
# ─────────────────────────────────────────────

# if __name__ == "__main__":
#     # ── Mock all pipeline outputs for a T-Test scenario ──
#     mock_profiler = {
#         "n_rows": 200, "n_cols": 2,
#         "continuous_columns": [{"column": "salary"}],
#         "categorical_columns": [{"column": "gender"}],
#         "warnings": [],
#     }
#     mock_preprocessor = {
#         "original_shape": [200, 2],
#         "final_shape": [197, 2],
#         "rows_dropped_total": 3,
#         "changes_summary": ["'salary': dropped 3 rows with missing values"],
#         "warnings": [],
#     }
#     mock_methodologist = {
#         "selected_test": "Independent Samples T-Test",
#         "selection_mode": "decided",
#         "dependent_variable": "salary",
#         "independent_variables": [],
#         "grouping_variable": "gender",
#         "n_rows": 197,
#         "reasoning": "T-Test selected for 2-group comparison.",
#         "original_query": "Does salary differ between genders?",
#     }
#     mock_checker = {
#         "total_assumptions": 4, "passed_count": 3,
#         "failed_count": 0, "warning_count": 1, "manual_count": 1,
#         "results": [
#             {"name": "normality", "status": "passed",
#              "plain_reason": "Shapiro-Wilk p=0.12 — normality met."},
#             {"name": "homogeneity_of_variance", "status": "passed",
#              "plain_reason": "Levene p=0.34 — equal variances."},
#             {"name": "no_significant_outliers", "status": "warning",
#              "plain_reason": "11 outliers (5.6%) — above 5% threshold."},
#             {"name": "independence_of_observations", "status": "manual",
#              "plain_reason": "User confirmed observations are independent."},
#         ],
#     }
#     mock_statistician = {
#         "test_name": "Independent Samples T-Test",
#         "test_family": "inference",
#         "n_observations": 197,
#         "correction_applied": None,
#         "inference_result": {
#             "test_name": "Independent Samples T-Test",
#             "statistic": -3.142,
#             "statistic_label": "t",
#             "p_value": 0.0018,
#             "alpha": 0.05,
#             "verdict": "significant",
#             "group_stats": {
#                 "M": {"mean": 57300, "std": 11800, "n": 110},
#                 "F": {"mean": 52100, "std": 10900, "n": 87},
#             },
#             "df": 195.0,
#             "effect_size": 0.44,
#             "effect_size_label": "Cohen's d",
#             "correction_applied": None,
#             "interpretation": (
#                 "Significant difference in salary between genders "
#                 "(t=-3.142, p=0.0018). Males earn more on average."
#             ),
#         },
#         "model_available_in_memory": False,
#         "columns_used": {"dependent": "salary", "grouping": "gender"},
#         "warnings": [],
#     }

#     print("\n" + "=" * 60)
#     print("  ARISTOSTAT — Final Report Agent")
#     print("=" * 60 + "\n")

#     output = run_final_report(
#         original_query="Does salary differ between genders?",
#         profiler_output=mock_profiler,
#         preprocessor_output=mock_preprocessor,
#         methodologist_output=mock_methodologist,
#         checker_output=mock_checker,
#         statistician_output=mock_statistician,
#         rectification_output=None,
#         critic_output=None,
#     )

#     print(output["final_response"])
#     print("\n" + "=" * 60)
#     print("docx generated:", output["report_output"].get("docx_generated"))
#     print("docx path:     ", output["report_output"].get("docx_path"))
#     print("caveats count: ", len(output["report_output"].get("caveats", [])))