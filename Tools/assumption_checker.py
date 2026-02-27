"""
FILE: tools/assumption_checker_tools.py
-----------------------------------------
LangChain tools exposed to the Assumption Checker ReAct agent.
"""

import json
import pandas as pd
from langchain_core.tools import tool

from Schemas.methodologist import MethodologistOutput
from core.assumption_engine import run_assumption_checks
from Utils.assumptions_requirements_registry import ASSUMPTION_REGISTRY


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_assumption_store: dict = {
    "methodologist_output": None,
    "cleaned_df":           None,
    "profiler_output":      None,
    "checker_output":       None,
}


def init_assumption_store(
    methodologist_output: dict,
    cleaned_df: pd.DataFrame,
    profiler_output: dict,
) -> None:
    """Called by run_assumption_checker() before invoking the agent."""
    _assumption_store["methodologist_output"] = methodologist_output
    _assumption_store["cleaned_df"]           = cleaned_df
    _assumption_store["profiler_output"]      = profiler_output
    _assumption_store["checker_output"]       = None


def get_assumption_store() -> dict:
    """Expose store to the agent runner."""
    return _assumption_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_test_and_columns() -> str:
    """
    Returns the selected test and confirmed column roles from the Methodologist.
    Call this first to know which test's assumptions to check and which
    columns are dependent, independent, and grouping variables.
    """
    output = _assumption_store.get("methodologist_output")
    if not output:
        return "ERROR: No methodologist output in store."

    return json.dumps({
        "selected_test":        output.get("selected_test"),
        "dependent_variable":   output.get("dependent_variable"),
        "independent_variables": output.get("independent_variables", []),
        "grouping_variable":    output.get("grouping_variable"),
        "selection_mode":       output.get("selection_mode"),
    }, indent=2)


@tool
def get_assumptions_for_test(test_name: str) -> str:
    """
    Returns the list of assumptions that will be checked for the given test.
    Useful for previewing what checks will run before executing them.
    Args:
        test_name: Canonical test name (e.g. 'One-Way ANOVA').
    """
    assumptions = ASSUMPTION_REGISTRY.get(test_name, [])
    if not assumptions:
        return f"No assumptions registered for test '{test_name}'."

    return json.dumps([
        {
            "name":         a["name"],
            "description":  a["description"],
            "check_method": a["check_method"],
        }
        for a in assumptions
    ], indent=2)


@tool
def run_all_assumption_checks() -> str:
    """
    Runs all pre-test assumption checks for the selected test.
    Each assumption is checked using the appropriate statistical test or heuristic.
    Assumptions that cannot be verified programmatically are flagged as MANUAL
    and require user confirmation.

    Returns the complete AssumptionCheckerOutput as a JSON string.
    Check the following fields in the result:
      - has_failures:             True if any assumption hard-failed
      - proceed_to_statistician:  True if safe to run the test
      - pending_manual_confirmations: assumptions needing user confirmation
      - results:                  Individual result for each assumption
    """
    methodologist_dict = _assumption_store.get("methodologist_output")
    cleaned_df         = _assumption_store.get("cleaned_df")
    profiler_output    = _assumption_store.get("profiler_output")

    if methodologist_dict is None:
        return "ERROR: No methodologist output in store."
    if cleaned_df is None:
        return "ERROR: No cleaned DataFrame in store."
    if profiler_output is None:
        return "ERROR: No profiler output in store."

    try:
        methodologist_output = MethodologistOutput(**methodologist_dict)
        checker_output = run_assumption_checks(
            methodologist_output=methodologist_output,
            cleaned_df=cleaned_df,
            profiler_output=profiler_output,
        )
        _assumption_store["checker_output"] = checker_output
        return checker_output.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Assumption checks failed — {str(e)}"


@tool
def get_failed_assumptions() -> str:
    """
    Returns only the failed assumptions from the last check run.
    Use this after run_all_assumption_checks to get a focused view
    of what went wrong and needs to be presented to the user.
    """
    output = _assumption_store.get("checker_output")
    if output is None:
        return "ERROR: No checker output available. Run run_all_assumption_checks first."

    failed = [
        {
            "name":        r.name,
            "description": r.description,
            "test_used":   r.test_used,
            "statistic":   r.statistic,
            "p_value":     r.p_value,
            "plain_reason": r.plain_reason,
        }
        for r in output.results
        if r.status.value == "failed"
    ]

    if not failed:
        return "No failed assumptions — all checks passed or are pending manual confirmation."

    return json.dumps(failed, indent=2)


@tool
def get_manual_confirmation_questions() -> str:
    """
    Returns the list of assumptions that require manual user confirmation,
    along with the specific question to ask the user for each.
    Use this to present manual checks to the user before proceeding.
    """
    output = _assumption_store.get("checker_output")
    if output is None:
        return "ERROR: No checker output available. Run run_all_assumption_checks first."

    manual = [
        {
            "name":     r.name,
            "question": r.manual_question,
        }
        for r in output.results
        if r.status.value == "manual"
    ]

    if not manual:
        return "No manual confirmations required."

    return json.dumps(manual, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

ASSUMPTION_CHECKER_TOOLS = [
    get_test_and_columns,
    get_assumptions_for_test,
    run_all_assumption_checks,
    get_failed_assumptions,
    get_manual_confirmation_questions,
]