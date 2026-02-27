"""
FILE: core/final_report_engine.py
-----------------------------------
Pure logic for assembling the Final Report from all upstream agent outputs.
No LangChain or LLM dependencies.

Responsibilities:
  1. Collects outputs from all agents in the pipeline
  2. Assembles structured report sections
  3. Builds caveats list from accepted violations, warnings, and post-test failures
  4. Generates markdown string for chat display
  5. Returns FinalReportOutput (docx generation is done in tools layer via Node.js)
"""

from Schemas.final_report import FinalReportOutput


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_dataset_summary(
    profiler_output: dict,
    preprocessor_output: dict,
) -> str:
    n_rows_orig = profiler_output.get("n_rows", "?")
    n_cols      = profiler_output.get("n_cols", "?")
    n_rows_final = preprocessor_output.get("final_shape", [n_rows_orig, n_cols])[0]
    rows_dropped = preprocessor_output.get("rows_dropped_total", 0)
    changes      = preprocessor_output.get("changes_summary", [])

    summary = f"Dataset: {n_rows_orig} rows × {n_cols} columns."
    if rows_dropped:
        summary += f" {rows_dropped} row(s) removed during cleaning ({n_rows_final} rows used for analysis)."
    if changes:
        summary += f" Cleaning applied: {'; '.join(changes[:3])}"
        if len(changes) > 3:
            summary += f" (and {len(changes) - 3} more)."
    return summary


def _build_test_selected_summary(
    methodologist_output: dict,
    rectification_output: dict | None,
) -> str:
    test_name = methodologist_output.get("selected_test", "Unknown test")
    mode      = methodologist_output.get("selection_mode", "decided")
    reasoning = methodologist_output.get("reasoning", "")

    # If rectification switched the test, note the new test
    if rectification_output and rectification_output.get("new_test"):
        new_test = rectification_output["new_test"]
        orig_test = methodologist_output.get("selected_test")
        return (
            f"Originally selected: {orig_test}. "
            f"Switched to {new_test} during rectification. "
            f"{reasoning}"
        )

    mode_notes = {
        "bypass":     "User explicitly requested this test.",
        "decided":    reasoning,
        "warned":     reasoning + " (minor compatibility warning noted).",
        "overridden": (
            f"User requested '{methodologist_output.get('user_requested_test')}' "
            f"but it was overridden — {methodologist_output.get('override_reason', '')} "
            f"{reasoning}"
        ),
    }
    return f"{test_name}. {mode_notes.get(mode, reasoning)}"


def _build_assumptions_summary(
    checker_output: dict,
) -> str:
    if not checker_output:
        return "Assumption checks were not run."

    total   = checker_output.get("total_assumptions", 0)
    passed  = checker_output.get("passed_count", 0)
    failed  = checker_output.get("failed_count", 0)
    manual  = checker_output.get("manual_count", 0)
    warning = checker_output.get("warning_count", 0)

    summary = f"{passed}/{total} assumptions passed."
    if failed:
        summary += f" {failed} failed (rectified or accepted)."
    if warning:
        summary += f" {warning} warnings noted."
    if manual:
        summary += f" {manual} confirmed manually by user."
    return summary


def _build_post_test_summary(
    critic_output: dict | None,
) -> str:
    if not critic_output:
        return ""
    if not critic_output.get("checks_applicable", False):
        return ""

    total   = critic_output.get("total_checks", 0)
    passed  = critic_output.get("passed_count", 0)
    failed  = critic_output.get("failed_count", 0)
    warning = critic_output.get("warning_count", 0)

    summary = f"Post-test model checks: {passed}/{total} passed."
    if failed:
        summary += f" {failed} failed (rectified or accepted)."
    if warning:
        summary += f" {warning} warnings noted."
    return summary


def _build_caveats(
    checker_output: dict,
    critic_output: dict | None,
    rectification_output: dict | None,
    preprocessor_output: dict,
) -> list[str]:
    caveats: list[str] = []

    # ── Accepted pre-test violations ──
    if rectification_output and rectification_output.get("user_accepted_violation"):
        for name in rectification_output.get("accepted_violation_names", []):
            caveats.append(
                f"Pre-test assumption '{name}' was not fully met and was acknowledged by the user. "
                f"Interpret results with appropriate caution."
            )

    # ── Assumption warnings ──
    for result in checker_output.get("results", []):
        if result.get("status") == "warning":
            caveats.append(f"Pre-test warning: {result.get('plain_reason', '')}")

    # ── Post-test failures accepted ──
    if critic_output and critic_output.get("has_failures") and critic_output.get("proceed_to_final_report"):
        for result in critic_output.get("results", []):
            if result.get("status") == "failed":
                caveats.append(
                    f"Post-test check '{result.get('name')}' failed: "
                    f"{result.get('plain_reason', '')} User chose to proceed."
                )

    # ── Preprocessor warnings ──
    for w in preprocessor_output.get("warnings", []):
        caveats.append(f"Data note: {w}")

    return caveats


def _build_key_statistic(statistician_output: dict) -> str:
    family = statistician_output.get("test_family", "")

    if family == "inference":
        r = statistician_output.get("inference_result", {})
        label   = r.get("statistic_label", "stat")
        stat    = r.get("statistic", "?")
        p       = r.get("p_value", "?")
        df      = r.get("df") or r.get("df_between")
        df_str  = f"({df})" if df is not None else ""
        return f"{label}{df_str} = {stat}, p = {p}"

    elif family == "regression":
        r = statistician_output.get("regression_result", {})
        return (
            f"R² = {r.get('r_squared', '?')}, "
            f"Adj. R² = {r.get('adj_r_squared', '?')}, "
            f"F = {r.get('f_statistic', '?')}, p = {r.get('f_p_value', '?')}"
        )

    elif family == "correlation":
        r = statistician_output.get("correlation_result", {})
        label = r.get("statistic_label", "r")
        stat  = r.get("statistic", "?")
        p     = r.get("p_value", "?")
        return f"{label} = {stat}, p = {p}"

    elif family == "dimensionality":
        r = statistician_output.get("dimensionality_result", {})
        return (
            f"{r.get('n_components_selected', '?')} components explain "
            f"{r.get('total_variance_explained', '?')}% of variance"
        )

    return "See results above."


def _build_verdict(statistician_output: dict) -> str:
    family = statistician_output.get("test_family", "")

    if family == "inference":
        r = statistician_output.get("inference_result", {})
        return r.get("interpretation", "")
    elif family == "regression":
        r = statistician_output.get("regression_result", {})
        return r.get("interpretation", "")
    elif family == "correlation":
        r = statistician_output.get("correlation_result", {})
        return r.get("interpretation", "")
    elif family == "dimensionality":
        r = statistician_output.get("dimensionality_result", {})
        return r.get("interpretation", "")
    return ""


def _build_effect_size(statistician_output: dict) -> str:
    family = statistician_output.get("test_family", "")
    if family == "inference":
        r = statistician_output.get("inference_result", {})
        if r.get("effect_size") and r.get("effect_size_label"):
            return f"{r['effect_size_label']} = {r['effect_size']}"
    elif family == "correlation":
        r = statistician_output.get("correlation_result", {})
        strength = r.get("correlation_strength", "")
        if strength:
            return f"{strength.capitalize()} correlation"
    return ""


# ─────────────────────────────────────────────
# MARKDOWN BUILDER
# ─────────────────────────────────────────────

def _build_markdown(report: FinalReportOutput) -> str:
    lines = []

    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"**Query:** {report.original_query}")
    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Dataset & Preparation")
    lines.append(report.dataset_summary)
    lines.append("")

    lines.append("## Test Selection")
    lines.append(report.test_selected)
    lines.append("")

    lines.append("## Assumption Checks")
    lines.append(report.assumptions_summary)
    if report.rectifications_applied:
        lines.append("")
        lines.append("**Rectifications applied:**")
        for r in report.rectifications_applied:
            lines.append(f"- {r}")
    lines.append("")

    if report.post_test_summary:
        lines.append("## Post-Test Model Checks")
        lines.append(report.post_test_summary)
        lines.append("")

    lines.append("## Results")
    lines.append(f"**{report.key_statistic}**")
    lines.append("")
    lines.append(report.verdict)
    if report.effect_size:
        lines.append("")
        lines.append(f"**Effect size:** {report.effect_size}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append(report.interpretation)
    lines.append("")

    if report.caveats:
        lines.append("## Caveats")
        for c in report.caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by Aristostat*")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN — ASSEMBLE REPORT
# ─────────────────────────────────────────────

def assemble_report(
    original_query: str,
    profiler_output: dict,
    preprocessor_output: dict,
    methodologist_output: dict,
    checker_output: dict,
    statistician_output: dict,
    rectification_output: dict | None = None,
    critic_output: dict | None = None,
    interpretation: str = "",
) -> FinalReportOutput:
    """
    Assembles all pipeline outputs into a structured FinalReportOutput.
    The interpretation field is filled in by the LLM agent (not computed here).
    docx_path is filled in by the tools layer after file generation.
    """
    test_name = statistician_output.get("test_name", "Statistical Analysis")

    dataset_summary     = _build_dataset_summary(profiler_output, preprocessor_output)
    test_selected       = _build_test_selected_summary(methodologist_output, rectification_output)
    assumptions_summary = _build_assumptions_summary(checker_output)
    post_test_summary   = _build_post_test_summary(critic_output)
    key_statistic       = _build_key_statistic(statistician_output)
    verdict             = _build_verdict(statistician_output)
    effect_size         = _build_effect_size(statistician_output)
    caveats             = _build_caveats(
        checker_output, critic_output, rectification_output, preprocessor_output
    )

    rectifications_applied: list[str] = []
    if rectification_output and rectification_output.get("chosen_solution"):
        sol = rectification_output["chosen_solution"]
        rectifications_applied.append(sol.get("description", ""))
    if rectification_output and rectification_output.get("applied_transforms"):
        for t in rectification_output["applied_transforms"]:
            rectifications_applied.append(
                f"{t.get('transform_type', 'Transform')} applied to '{t.get('column', '?')}'."
            )

    report = FinalReportOutput(
        title=f"Statistical Analysis Report — {test_name}",
        original_query=original_query,
        dataset_summary=dataset_summary,
        test_selected=test_selected,
        assumptions_summary=assumptions_summary,
        rectifications_applied=rectifications_applied,
        post_test_summary=post_test_summary,
        key_statistic=key_statistic,
        verdict=verdict,
        effect_size=effect_size,
        interpretation=interpretation,
        caveats=caveats,
    )

    report.markdown_report = _build_markdown(report)
    return report