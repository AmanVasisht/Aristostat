"""
FILE: tools/methodologist_tools.py
------------------------------------
LangChain tools exposed to the Methodologist ReAct agent.
The LLM uses these tools to access intent and profiler data,
run the decision engine, and produce a MethodologistOutput.
"""

import json
from langchain_core.tools import tool

from schemas.intent_schema import IntentOutput
from core.methodologist_engine import build_methodologist_output


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_methodologist_store: dict = {
    "intent_output":        None,
    "profiler_output":      None,
    "methodologist_output": None,
}


def init_methodologist_store(intent_output: dict, profiler_output: dict) -> None:
    """Called by run_methodologist() before invoking the agent."""
    _methodologist_store["intent_output"]        = intent_output
    _methodologist_store["profiler_output"]      = profiler_output
    _methodologist_store["methodologist_output"] = None


def get_methodologist_store() -> dict:
    """Expose store to the agent runner for retrieving final MethodologistOutput."""
    return _methodologist_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_intent_summary() -> str:
    """
    Returns a summary of the IntentOutput from the Intent Interpreter.
    Includes intent type, analysis goal, confidence, requested test (if any),
    columns and their roles, and whether the Methodologist should be bypassed.
    Call this first to understand what the user wants.
    """
    intent = _methodologist_store.get("intent_output")
    if not intent:
        return "ERROR: No intent output available in store."

    summary = {
        "intent_type":          intent.get("intent_type"),
        "analysis_goal":        intent.get("analysis_goal"),
        "confidence":           intent.get("confidence"),
        "methodologist_bypass": intent.get("methodologist_bypass"),
        "requested_test":       intent.get("requested_test"),
        "columns": [
            {
                "name":  c.get("name"),
                "role":  c.get("role"),
                "dtype": c.get("dtype_from_profiler"),
            }
            for c in intent.get("columns", [])
        ],
        "original_query": intent.get("original_query"),
    }
    return json.dumps(summary, indent=2)


@tool
def get_profiler_summary() -> str:
    """
    Returns key facts from the Data Profiler output needed for test selection —
    number of rows, column names and types, and any profiler warnings.
    Useful for checking sample size and column type distribution.
    """
    profiler = _methodologist_store.get("profiler_output")
    if not profiler:
        return "ERROR: No profiler output available in store."

    summary = {
        "n_rows":              profiler.get("n_rows"),
        "n_cols":              profiler.get("n_cols"),
        "continuous_columns":  [c["column"] for c in profiler.get("continuous_columns", [])],
        "categorical_columns": [c["column"] for c in profiler.get("categorical_columns", [])],
        "profiler_warnings":   profiler.get("warnings", []),
    }
    return json.dumps(summary, indent=2)


@tool
def get_grouping_column_details(column_name: str) -> str:
    """
    Returns cardinality and class info for a categorical column.
    Use this to check how many groups exist in a grouping variable —
    critical for choosing between T-Test (2 groups), ANOVA (3+ groups),
    and their non-parametric equivalents.
    Args:
        column_name: Name of the categorical column to inspect.
    """
    profiler = _methodologist_store.get("profiler_output")
    if not profiler:
        return "ERROR: No profiler output available."

    for col in profiler.get("categorical_columns", []):
        if col["column"] == column_name:
            return json.dumps({
                "column":          col["column"],
                "cardinality":     col.get("cardinality"),
                "mode":            col.get("mode"),
                "mode_freq":       col.get("mode_frequency"),
                "class_imbalance": col.get("class_imbalance_flag"),
            }, indent=2)

    return f"ERROR: Column '{column_name}' not found in categorical columns."


@tool
def select_test() -> str:
    """
    Runs the Methodologist decision engine on the current intent and profiler output.

    Handles three bypass outcomes:
      - BYPASS:     user named a test, column types are compatible — passed through.
      - WARNED:     user named a test, minor issue found (e.g. small sample size) —
                    warning shown but their test is used.
      - OVERRIDDEN: user named a test but it is clearly wrong (type mismatch,
                    missing grouping variable, incompatible group count) —
                    the correct test is selected via decision tree instead.

    And one decision tree outcome:
      - DECIDED:    methodologist picked the test from scratch.

    Returns the complete MethodologistOutput as a JSON string.
    Check selection_mode, override_reason, and mismatch_warning fields in the result.
    """
    intent_dict = _methodologist_store.get("intent_output")
    profiler    = _methodologist_store.get("profiler_output")

    if not intent_dict:
        return "ERROR: No intent output in store."
    if not profiler:
        return "ERROR: No profiler output in store."

    try:
        intent = IntentOutput(**intent_dict)
        result = build_methodologist_output(
            intent=intent,
            profiler_output=profiler,
        )
        _methodologist_store["methodologist_output"] = result
        return result.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Methodologist engine failed — {str(e)}"


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

METHODOLOGIST_TOOLS = [
    get_intent_summary,
    get_profiler_summary,
    get_grouping_column_details,
    select_test,
]