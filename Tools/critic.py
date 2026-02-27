"""
FILE: tools/model_critic_tools.py
------------------------------------
LangChain tools exposed to the Model Critic ReAct agent.
The fitted model object is received directly in memory from the Statistician.
"""

import json
import pandas as pd
from langchain_core.tools import tool

from core.critic_engine import run_post_test_checks


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_model_critic_store: dict = {
    "statistician_output":  None,
    "fitted_model":         None,   # passed directly from Statistician in memory
    "cleaned_df":           None,
    "methodologist_output": None,
    "critic_output":        None,
}


def init_model_critic_store(
    statistician_output: dict,
    fitted_model: object | None,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
) -> None:
    """Called by run_model_critic() before invoking the agent."""
    _model_critic_store["statistician_output"]  = statistician_output
    _model_critic_store["fitted_model"]         = fitted_model
    _model_critic_store["cleaned_df"]           = cleaned_df
    _model_critic_store["methodologist_output"] = methodologist_output
    _model_critic_store["critic_output"]        = None


def get_model_critic_store() -> dict:
    """Expose store to the agent runner."""
    return _model_critic_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_model_context() -> str:
    """
    Returns context about the fitted model — test name, test family,
    and whether post-test checks are applicable.
    Call this first to understand what checks need to run.
    Checks only apply to regression tests (SLR, MLR, Ridge, Lasso).
    For inference, correlation, and dimensionality tests, checks are skipped.
    """
    stat_out = _model_critic_store.get("statistician_output", {})
    meth_out = _model_critic_store.get("methodologist_output", {})

    return json.dumps({
        "test_name":            stat_out.get("test_name"),
        "test_family":          stat_out.get("test_family"),
        "model_in_memory":      _model_critic_store.get("fitted_model") is not None,
        "n_observations":       stat_out.get("n_observations"),
        "dependent_variable":   meth_out.get("dependent_variable"),
        "independent_variables": meth_out.get("independent_variables", []),
        "correction_applied":   stat_out.get("correction_applied"),
    }, indent=2)


@tool
def run_post_test_assumption_checks() -> str:
    """
    Runs all post-test assumption checks for the fitted model.
    Only applies to regression tests — for other test families,
    returns immediately with checks_applicable=False.

    Post-test checks for regression:
      - Normality of residuals (Shapiro-Wilk / D'Agostino)
      - Homoscedasticity of residuals (Breusch-Pagan)
      - No autocorrelation (Durbin-Watson — should be close to 2)
      - No influential points (Cook's Distance > 4/n)

    Returns the complete ModelCriticOutput as a JSON string.
    Check the following fields:
      - checks_applicable:      False if test family has no post-test checks
      - has_failures:           True if any check hard-failed
      - proceed_to_final_report: True if safe to proceed
    """
    stat_out  = _model_critic_store.get("statistician_output", {})
    fitted    = _model_critic_store.get("fitted_model")
    df        = _model_critic_store.get("cleaned_df")
    meth_out  = _model_critic_store.get("methodologist_output", {})

    if df is None:
        return "ERROR: No DataFrame in store."

    test_name    = stat_out.get("test_name", "Unknown")
    test_family  = stat_out.get("test_family", "inference")
    dep_var      = meth_out.get("dependent_variable")
    ind_vars     = meth_out.get("independent_variables", [])

    try:
        output = run_post_test_checks(
            test_family=test_family,
            fitted_model=fitted,
            df=df,
            dependent_var=dep_var,
            independent_vars=ind_vars,
            test_name=test_name,
        )
        _model_critic_store["critic_output"] = output
        return output.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Post-test checks failed — {str(e)}"


@tool
def get_failed_post_test_checks() -> str:
    """
    Returns only the failed post-test checks for focused presentation to the user.
    Use this after run_post_test_assumption_checks to get a clean list of failures.
    """
    output = _model_critic_store.get("critic_output")
    if output is None:
        return "ERROR: No critic output. Run run_post_test_assumption_checks first."

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
        return "No failed post-test checks — all checks passed or were not applicable."

    return json.dumps(failed, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

MODEL_CRITIC_TOOLS = [
    get_model_context,
    run_post_test_assumption_checks,
    get_failed_post_test_checks,
]