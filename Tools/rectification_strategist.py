"""
FILE: tools/rectification_tools.py
------------------------------------
LangChain tools exposed to the Rectification Strategist ReAct agent.
Used for both pre-test (from Assumption Checker) and
post-test (from Model Critic) failures — distinguished by phase flag.

Tools:
  get_failure_context        — basic failure info + attempt count
  get_violation_details      — rich context: VIF scores, statistics, dataset info for REASONING
  get_proposed_solutions     — fetch solutions from RECTIFICATION_REGISTRY (deterministic)
  apply_chosen_solution      — apply a solution_id to the data
  resolve_columns_to_drop    — parse free-text column names from user input
  accept_violation_and_proceed — user chose to proceed despite failures
  check_attempt_limit        — check if max attempts reached
"""

import json
import re
import pandas as pd
from langchain_core.tools import tool

from Schemas.rectification_strategist import RectificationPhase
from core.rectification_engine import build_rectification_output, get_proposals_for_failures


# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_rectification_store: dict = {
    "failed_assumptions":    None,
    "phase":                 None,
    "cleaned_df":            None,
    "dependent_var":         None,
    "independent_vars":      None,
    "methodologist_output":  None,
    "checker_output":        None,   # NEW — full checker output for VIF reasoning
    "critic_output":         None,   # NEW — full critic output for post-test reasoning
    "rectification_attempt": 1,
    "max_attempts":          3,
    "rectification_output":  None,
}


def init_rectification_store(
    failed_assumptions: list[str],
    phase: str,
    cleaned_df: pd.DataFrame,
    methodologist_output: dict,
    rectification_attempt: int = 1,
    max_attempts: int = 3,
    checker_output: dict | None = None,
    critic_output: dict | None = None,
) -> None:
    """Called by run_rectification_strategist() before invoking the agent."""
    _rectification_store["failed_assumptions"]    = failed_assumptions
    _rectification_store["phase"]                 = phase
    _rectification_store["cleaned_df"]            = cleaned_df
    _rectification_store["dependent_var"]         = methodologist_output.get("dependent_variable")
    _rectification_store["independent_vars"]      = methodologist_output.get("independent_variables", []).copy()
    _rectification_store["methodologist_output"]  = methodologist_output
    _rectification_store["checker_output"]        = checker_output or {}
    _rectification_store["critic_output"]         = critic_output or {}
    _rectification_store["rectification_attempt"] = rectification_attempt
    _rectification_store["max_attempts"]          = max_attempts
    _rectification_store["rectification_output"]  = None


def get_rectification_store() -> dict:
    return _rectification_store


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def get_failure_context() -> str:
    """
    Returns basic context about the current failure situation:
    which assumptions failed, the phase (pre_test or post_test),
    current rectification attempt number, max allowed attempts,
    and the column roles in the model.
    Call this first.
    """
    store = _rectification_store
    return json.dumps({
        "failed_assumptions":    store.get("failed_assumptions", []),
        "phase":                 store.get("phase"),
        "rectification_attempt": store.get("rectification_attempt"),
        "max_attempts":          store.get("max_attempts"),
        "test_name":             store.get("methodologist_output", {}).get("selected_test"),
        "dependent_variable":    store.get("dependent_var"),
        "independent_variables": store.get("independent_vars", []),
        "original_query":        store.get("methodologist_output", {}).get("original_query", ""),
        "n_rows":                store.get("methodologist_output", {}).get("n_rows"),
    }, indent=2)


@tool
def get_violation_details() -> str:
    """
    Returns rich, detailed information about each failed assumption for reasoning.
    Includes exact VIF scores per variable, test statistics, p-values,
    and dataset context (sample size, predictor count).

    Use this to reason about WHICH solution best fits this specific situation.
    For example: if two variables have nearly identical VIF scores, they are
    likely measuring the same construct and one should be dropped.
    If VIF is only slightly above 10, Ridge may be sufficient.
    If n is small and predictors are many, dropping is better than regularisation.
    """
    store           = _rectification_store
    checker_output  = store.get("checker_output", {})
    critic_output   = store.get("critic_output", {})
    methodologist   = store.get("methodologist_output", {})
    ind_vars        = store.get("independent_vars", [])
    n_rows          = methodologist.get("n_rows") or "unknown"

    details = {
        "dataset_context": {
            "n_rows":           n_rows,
            "n_predictors":     len(ind_vars),
            "independent_vars": ind_vars,
            "test_name":        methodologist.get("selected_test"),
            "user_goal":        methodologist.get("original_query", ""),
        },
        "violations": [],
    }

    all_results = (
        checker_output.get("results", []) +
        critic_output.get("results", [])
    )

    for r in all_results:
        status = r.get("status", "")
        if status not in ("failed", "warning"):
            continue

        violation = {
            "assumption":   r.get("name"),
            "status":       status,
            "test_used":    r.get("test_used"),
            "statistic":    r.get("statistic"),
            "p_value":      r.get("p_value"),
            "plain_reason": r.get("plain_reason", ""),
        }

        # ── For multicollinearity: parse individual VIF scores ──
        if "multicollinear" in (r.get("name") or "").lower() or "vif" in r.get("plain_reason", "").lower():
            vif_scores = {}
            reason = r.get("plain_reason", "")
            # Match patterns like "Age: 27.14 (≥10)" or "Age: 1.08"
            for match in re.finditer(r"([\w\s]+?):\s*([\d.]+(?:,[\d.]+)?)\s*(?:\([^)]*\))?", reason):
                var_name = match.group(1).strip()
                vif_val  = match.group(2).replace(",", ".")
                try:
                    vif_scores[var_name] = float(vif_val)
                except ValueError:
                    pass
            if vif_scores:
                violation["vif_scores_per_variable"] = vif_scores
                # Identify the worst offenders
                high_vif = {k: v for k, v in vif_scores.items() if v >= 10}
                violation["high_vif_variables"]      = list(high_vif.keys())
                violation["max_vif"]                 = max(vif_scores.values()) if vif_scores else None

        details["violations"].append(violation)

    # ── Add reasoning hints based on context ──
    hints = []
    n = n_rows if isinstance(n_rows, int) else 0
    p = len(ind_vars)

    if n > 0 and p > 0:
        if n < p * 10:
            hints.append(
                f"Sample size warning: n={n}, p={p}. Rule of thumb requires n >= {p*10}. "
                f"Fewer predictors = better model stability."
            )
        if n <= p + 1:
            hints.append(
                f"CRITICAL: n={n} <= p+1={p+1}. OLS will overfit. "
                f"Must either drop predictors or use regularisation (Ridge/Lasso)."
            )

    if hints:
        details["reasoning_hints"] = hints

    return json.dumps(details, indent=2)


@tool
def get_proposed_solutions() -> str:
    """
    Returns the list of proposed solutions for the failed assumptions,
    fetched directly from the RECTIFICATION_REGISTRY.
    Solutions are ranked: corrections first, then test switches, transforms, drops.

    Use get_violation_details BEFORE calling this so you can reason about
    which solution best fits the specific situation when presenting to the user.
    """
    failed    = _rectification_store.get("failed_assumptions", [])
    phase_str = _rectification_store.get("phase", "pre_test")

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    proposals = get_proposals_for_failures(failed, phase)

    if not proposals:
        return json.dumps({
            "message":     "No automated solutions available for these failures.",
            "suggestions": "Consider collecting more data or proceeding with acknowledged violations."
        }, indent=2)

    return json.dumps([
        {
            "option_number": i + 1,
            "solution_id":   p.solution_id,
            "description":   p.description,
            "action_type":   p.action_type,
            "next_step":     p.next_step,
        }
        for i, p in enumerate(proposals)
    ], indent=2)


@tool
def resolve_columns_to_drop(user_text: str) -> str:
    """
    Parses free-text user input to identify which columns they want to drop.
    Matches by meaning, not exact spelling.

    Use this when the user describes a column drop in natural language,
    e.g. "drop age", "remove age and years of experience", "get rid of the age variable".

    Args:
        user_text: The user's free-text description of which variable(s) to drop.

    Returns:
        JSON with matched column names from the available independent variables.
    """
    ind_vars   = _rectification_store.get("independent_vars", [])
    user_lower = user_text.lower()

    # ── Deterministic pass first: case-insensitive exact and word match ──
    matched = []
    for col in ind_vars:
        col_lower = col.lower()
        col_words = col_lower.split()
        # Exact match or all words appear in user text
        if col_lower in user_lower or all(w in user_lower for w in col_words):
            matched.append(col)

    # ── Also match common abbreviations ──
    abbreviations = {
        "yoe": "Years of Experience",
        "years exp": "Years of Experience",
        "exp": "Years of Experience",
        "yrs": "Years of Experience",
    }
    for abbrev, full_name in abbreviations.items():
        if abbrev in user_lower and full_name in ind_vars and full_name not in matched:
            matched.append(full_name)

    return json.dumps({
        "user_text":          user_text,
        "available_columns":  ind_vars,
        "matched_columns":    matched,
        "match_count":        len(matched),
        "message": (
            f"Identified {len(matched)} column(s) to drop: {matched}"
            if matched else
            "Could not identify any columns from the user's input. Ask for clarification."
        )
    }, indent=2)


@tool
def apply_chosen_solution(solution_id: str, columns_to_drop: str = "") -> str:
    """
    Applies the chosen rectification solution to the data.

    For transforms: modifies the DataFrame in the store.
    For test switches: records the new test name.
    For corrections: records the correction type.
    For drop_variable: drops the specified column(s) from the DataFrame.

    Returns the RectificationOutput as JSON.
    Key fields to report to user:
      - next_step: "assumption_checker" or "statistician"
      - new_test: set if test was switched
      - correction_type: set if correction applied
      - applied_transforms: list of data changes

    Args:
        solution_id:      The solution_id from get_proposed_solutions().
        columns_to_drop:  Comma-separated column name(s) to drop — ONLY needed
                          when solution_id is "multicollinearity_drop_variable".
                          Use resolve_columns_to_drop first to identify column names.
    """
    failed       = _rectification_store.get("failed_assumptions", [])
    phase_str    = _rectification_store.get("phase", "pre_test")
    df           = _rectification_store.get("cleaned_df")
    dep_var      = _rectification_store.get("dependent_var")
    ind_vars     = _rectification_store.get("independent_vars", []).copy()
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)

    if df is None:
        return "ERROR: No DataFrame in store."

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    # ── Handle drop_variable — apply drop before calling engine ──
    if solution_id == "multicollinearity_drop_variable":
        if not columns_to_drop:
            return (
                "ERROR: columns_to_drop is required for drop_variable solution. "
                "Call resolve_columns_to_drop first to identify which columns the user wants to drop, "
                "then pass them here as a comma-separated string."
            )

        drop_list   = [c.strip() for c in columns_to_drop.split(",") if c.strip()]
        cols_in_df  = [c for c in drop_list if c in df.columns]
        invalid     = [c for c in drop_list if c not in df.columns]

        if invalid:
            available = list(df.columns)
            return (
                f"ERROR: Columns not found in DataFrame: {invalid}. "
                f"Available columns: {available}. "
                f"Call resolve_columns_to_drop again to get the correct column names."
            )

        # Drop from df and update ind_vars
        df       = df.drop(columns=cols_in_df)
        ind_vars = [v for v in ind_vars if v not in drop_list]

        # Update store
        _rectification_store["cleaned_df"]       = df
        _rectification_store["independent_vars"] = ind_vars

        # Inject column info into the solution for engine logging
        # (engine will find and log it via action_details)
        from core.rectification_engine import get_proposals_for_failures
        proposals = get_proposals_for_failures(failed, phase)
        for p in proposals:
            if p.solution_id == "multicollinearity_drop_variable":
                p.action_details["column"] = ", ".join(cols_in_df)

    try:
        rectified_df, output = build_rectification_output(
            failed_assumptions=failed,
            phase=phase,
            chosen_solution_id=solution_id,
            df=df,
            dependent_var=dep_var,
            independent_vars=ind_vars,
            rectification_attempt=attempt,
            max_attempts=max_attempts,
        )
        _rectification_store["cleaned_df"]          = rectified_df
        _rectification_store["independent_vars"]    = ind_vars   # updated if drop happened
        _rectification_store["rectification_output"] = output

        result = output.model_dump()
        # Add a plain English summary for the agent to report
        result["_summary"] = (
            f"Dropped: {cols_in_df}. Remaining predictors: {ind_vars}. "
            f"Re-running assumption checks."
            if solution_id == "multicollinearity_drop_variable" else
            f"Solution applied. Next step: {output.next_step.value}."
        )
        return json.dumps(result, default=str, indent=2)

    except Exception as e:
        return f"ERROR: Failed to apply solution — {str(e)}"


@tool
def accept_violation_and_proceed(violation_names: str) -> str:
    """
    Called when the user chooses to proceed despite assumption failures.
    Records the accepted violations and routes directly to the Statistician.

    Args:
        violation_names: Comma-separated assumption names the user is accepting.
                         e.g. "normality,homogeneity_of_variance"
    """
    failed       = _rectification_store.get("failed_assumptions", [])
    phase_str    = _rectification_store.get("phase", "pre_test")
    df           = _rectification_store.get("cleaned_df")
    dep_var      = _rectification_store.get("dependent_var")
    ind_vars     = _rectification_store.get("independent_vars", [])
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)
    accepted     = [v.strip() for v in violation_names.split(",")]

    try:
        phase = RectificationPhase(phase_str)
    except ValueError:
        phase = RectificationPhase.PRE_TEST

    _, output = build_rectification_output(
        failed_assumptions=failed,
        phase=phase,
        chosen_solution_id=None,
        df=df,
        dependent_var=dep_var,
        independent_vars=ind_vars,
        rectification_attempt=attempt,
        max_attempts=max_attempts,
        user_accepted_violation=True,
        accepted_violation_names=accepted,
    )
    _rectification_store["rectification_output"] = output
    return output.model_dump_json(indent=2)


@tool
def check_attempt_limit() -> str:
    """
    Returns whether the maximum rectification attempt limit has been reached.
    If exceeded, the user must accept violations or stop.
    """
    attempt      = _rectification_store.get("rectification_attempt", 1)
    max_attempts = _rectification_store.get("max_attempts", 3)
    exceeded     = attempt > max_attempts

    return json.dumps({
        "current_attempt": attempt,
        "max_attempts":    max_attempts,
        "limit_exceeded":  exceeded,
        "message": (
            f"Maximum rectification attempts ({max_attempts}) reached."
            if exceeded else
            f"Attempt {attempt} of {max_attempts}."
        )
    }, indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST
# ─────────────────────────────────────────────

RECTIFICATION_TOOLS = [
    get_failure_context,
    get_violation_details,
    get_proposed_solutions,
    resolve_columns_to_drop,
    apply_chosen_solution,
    accept_violation_and_proceed,
    check_attempt_limit,
]