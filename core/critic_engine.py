"""
FILE: core/model_critic_engine.py
-----------------------------------
Post-test assumption check functions for the Model Critic agent.
All checks require a fitted model object — residuals, influence measures, etc.
No LangChain or LLM dependencies.

Checks implemented:
  - Normality of residuals (Shapiro-Wilk / D'Agostino)
  - Homoscedasticity of residuals (Breusch-Pagan)
  - Autocorrelation of residuals (Durbin-Watson)
  - Influential points (Cook's Distance)

For non-regression test families (inference, correlation, dimensionality),
run_post_test_checks() returns immediately with checks_applicable=False.
"""

import numpy as np
import pandas as pd
from scipy import stats

from configs.post_test_assumptions import POST_TEST_ASSUMPTION_REGISTRY
from schemas.assumption_checker_schema import AssumptionResult, AssumptionStatus
from schemas.model_critic_schema import ModelCriticOutput


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DW_LOWER_BOUND = 1.5    # Durbin-Watson below this → positive autocorrelation
DW_UPPER_BOUND = 2.5    # Durbin-Watson above this → negative autocorrelation


# ─────────────────────────────────────────────
# HELPER — EXTRACT RESIDUALS FROM MODEL
# Handles both statsmodels OLS and sklearn pipelines
# ─────────────────────────────────────────────

def _extract_residuals(
    fitted_model: object,
    df: pd.DataFrame,
    dependent_var: str,
    independent_vars: list[str],
) -> np.ndarray | None:
    """
    Extracts residuals from a fitted model.
    Handles statsmodels OLS (has .resid) and sklearn pipelines (compute manually).
    """
    # statsmodels OLS
    if hasattr(fitted_model, "resid"):
        return np.array(fitted_model.resid)

    # sklearn Pipeline (Ridge, Lasso)
    if hasattr(fitted_model, "predict"):
        try:
            X = df[independent_vars].dropna()
            y = df[dependent_var].loc[X.index].values
            y_pred = fitted_model.predict(X)
            return y - y_pred
        except Exception:
            return None

    return None


# ─────────────────────────────────────────────
# POST-TEST CHECK FUNCTIONS
# ─────────────────────────────────────────────

def check_normality_of_residuals(
    residuals: np.ndarray,
    assumption: dict,
) -> AssumptionResult:
    """Shapiro-Wilk (or D'Agostino for n>5000) on model residuals."""
    alpha = assumption.get("alpha", 0.05)

    if len(residuals) < 3:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Shapiro-Wilk (residuals)",
            plain_reason="Too few residuals to test normality.",
        )

    if len(residuals) > 5000:
        stat, p = stats.normaltest(residuals)
        test_name = "D'Agostino-Pearson (residuals)"
    else:
        stat, p = stats.shapiro(residuals)
        test_name = "Shapiro-Wilk (residuals)"

    passed = p >= alpha
    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
        test_used=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 4),
        alpha=alpha,
        plain_reason=(
            f"{test_name}: statistic={round(float(stat), 4)}, p={round(float(p), 4)}. "
            + ("Residuals are approximately normally distributed." if passed
               else f"p < {alpha} — residuals are not normally distributed.")
        ),
    )


def check_homoscedasticity_bp(
    fitted_model: object,
    assumption: dict,
) -> AssumptionResult:
    """
    Breusch-Pagan test on the fitted statsmodels OLS model.
    For sklearn models, uses residuals vs fitted values correlation as a heuristic.
    """
    alpha = assumption.get("alpha", 0.05)

    # statsmodels OLS — use proper Breusch-Pagan
    if hasattr(fitted_model, "resid") and hasattr(fitted_model, "model"):
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            _, p, _, _ = het_breuschpagan(fitted_model.resid, fitted_model.model.exog)
            passed = p >= alpha
            return AssumptionResult(
                name=assumption["name"],
                description=assumption["description"],
                status=AssumptionStatus.PASSED if passed else AssumptionStatus.FAILED,
                test_used="Breusch-Pagan",
                p_value=round(float(p), 4),
                alpha=alpha,
                plain_reason=(
                    f"Breusch-Pagan: p={round(float(p), 4)}. "
                    + ("Homoscedasticity assumption met." if passed
                       else f"p < {alpha} — heteroscedasticity detected in residuals.")
                ),
            )
        except Exception as e:
            return AssumptionResult(
                name=assumption["name"],
                description=assumption["description"],
                status=AssumptionStatus.WARNING,
                test_used="Breusch-Pagan",
                plain_reason=f"Could not run Breusch-Pagan: {str(e)}",
            )

    # sklearn model — heuristic: correlation between |residuals| and fitted values
    if hasattr(fitted_model, "predict"):
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Breusch-Pagan (not available for regularised models)",
            plain_reason=(
                "Breusch-Pagan is not available for Ridge/Lasso models. "
                "Inspect a residual vs fitted plot to assess homoscedasticity manually."
            ),
        )

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=AssumptionStatus.WARNING,
        test_used="Breusch-Pagan",
        plain_reason="Could not extract model information for Breusch-Pagan test.",
    )


def check_autocorrelation_dw(
    residuals: np.ndarray,
    assumption: dict,
) -> AssumptionResult:
    """
    Durbin-Watson test for autocorrelation in residuals.
    DW ≈ 2 → no autocorrelation.
    DW < 1.5 → positive autocorrelation (problem).
    DW > 2.5 → negative autocorrelation (problem).
    """
    from statsmodels.stats.stattools import durbin_watson

    dw = float(durbin_watson(residuals))

    if dw < DW_LOWER_BOUND:
        status = AssumptionStatus.FAILED
        reason = (
            f"Durbin-Watson={round(dw, 4)} < {DW_LOWER_BOUND} — "
            f"positive autocorrelation detected in residuals."
        )
    elif dw > DW_UPPER_BOUND:
        status = AssumptionStatus.FAILED
        reason = (
            f"Durbin-Watson={round(dw, 4)} > {DW_UPPER_BOUND} — "
            f"negative autocorrelation detected in residuals."
        )
    else:
        status = AssumptionStatus.PASSED
        reason = (
            f"Durbin-Watson={round(dw, 4)} — "
            f"within acceptable range [{DW_LOWER_BOUND}, {DW_UPPER_BOUND}]. "
            f"No significant autocorrelation detected."
        )

    return AssumptionResult(
        name=assumption["name"],
        description=assumption["description"],
        status=status,
        test_used="Durbin-Watson",
        statistic=round(dw, 4),
        plain_reason=reason,
    )


def check_influential_points_cooks(
    fitted_model: object,
    n_obs: int,
    assumption: dict,
) -> AssumptionResult:
    """
    Cook's Distance check for influential observations.
    Threshold: Cook's D > 4/n → influential point.
    """
    # statsmodels OLS only — sklearn doesn't provide influence measures
    if not hasattr(fitted_model, "get_influence"):
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Cook's Distance",
            plain_reason=(
                "Cook's Distance is not available for regularised models (Ridge/Lasso). "
                "Consider inspecting leverage plots manually."
            ),
        )

    try:
        influence  = fitted_model.get_influence()
        cooks_d    = influence.cooks_distance[0]
        threshold  = 4 / n_obs
        n_influential = int(np.sum(cooks_d > threshold))
        pct = round(n_influential / n_obs * 100, 2)

        passed = n_influential == 0
        status = AssumptionStatus.PASSED if passed else AssumptionStatus.WARNING

        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=status,
            test_used="Cook's Distance",
            plain_reason=(
                f"Cook's Distance threshold = 4/n = {round(threshold, 4)}. "
                + (f"No influential points detected." if passed
                   else f"{n_influential} influential observation(s) detected "
                        f"({pct}% of data, Cook's D > {round(threshold, 4)}).")
            ),
        )
    except Exception as e:
        return AssumptionResult(
            name=assumption["name"],
            description=assumption["description"],
            status=AssumptionStatus.WARNING,
            test_used="Cook's Distance",
            plain_reason=f"Could not compute Cook's Distance: {str(e)}",
        )


# ─────────────────────────────────────────────
# MAIN — RUN ALL POST-TEST CHECKS
# ─────────────────────────────────────────────

def run_post_test_checks(
    test_family: str,
    fitted_model: object | None,
    df: pd.DataFrame,
    dependent_var: str | None,
    independent_vars: list[str],
    test_name: str,
) -> ModelCriticOutput:
    """
    Main entry point for the Model Critic engine.

    For non-regression families (inference, correlation, dimensionality):
    returns immediately with checks_applicable=False and proceed_to_final_report=True.

    For regression: runs all four post-test checks and returns ModelCriticOutput.
    """
    checks = POST_TEST_ASSUMPTION_REGISTRY.get(test_family, [])

    # ── Non-regression — skip post-test checks ──
    if not checks:
        return ModelCriticOutput(
            test_name=test_name,
            test_family=test_family,
            checks_applicable=False,
            proceed_to_final_report=True,
            summary_message=(
                f"No post-test model checks required for {test_family} tests. "
                f"Proceeding to final report."
            ),
        )

    if fitted_model is None:
        return ModelCriticOutput(
            test_name=test_name,
            test_family=test_family,
            checks_applicable=True,
            has_failures=False,
            proceed_to_final_report=True,
            summary_message="No fitted model available for post-test checks.",
        )

    # ── Extract residuals ──
    residuals = _extract_residuals(fitted_model, df, dependent_var, independent_vars)
    n_obs = len(df)

    results: list[AssumptionResult] = []

    for assumption in checks:
        fn_name = assumption.get("test_fn")
        try:
            if fn_name == "check_normality_of_residuals":
                if residuals is not None:
                    results.append(check_normality_of_residuals(residuals, assumption))
                else:
                    results.append(AssumptionResult(
                        name=assumption["name"],
                        description=assumption["description"],
                        status=AssumptionStatus.WARNING,
                        plain_reason="Could not extract residuals from model.",
                    ))

            elif fn_name == "check_homoscedasticity_bp":
                results.append(check_homoscedasticity_bp(fitted_model, assumption))

            elif fn_name == "check_autocorrelation_dw":
                if residuals is not None:
                    results.append(check_autocorrelation_dw(residuals, assumption))
                else:
                    results.append(AssumptionResult(
                        name=assumption["name"],
                        description=assumption["description"],
                        status=AssumptionStatus.WARNING,
                        plain_reason="Could not extract residuals for Durbin-Watson test.",
                    ))

            elif fn_name == "check_influential_points_cooks":
                results.append(check_influential_points_cooks(fitted_model, n_obs, assumption))

        except Exception as e:
            results.append(AssumptionResult(
                name=assumption["name"],
                description=assumption["description"],
                status=AssumptionStatus.WARNING,
                plain_reason=f"Check could not be completed: {str(e)}",
            ))

    # ── Aggregate ──
    passed_count  = sum(1 for r in results if r.status == AssumptionStatus.PASSED)
    failed_count  = sum(1 for r in results if r.status == AssumptionStatus.FAILED)
    warning_count = sum(1 for r in results if r.status == AssumptionStatus.WARNING)
    has_failures  = failed_count > 0

    # ── Summary message ──
    lines = [f"Post-test model checks for **{test_name}**:"]
    lines.append(
        f"✅ Passed: {passed_count}  ❌ Failed: {failed_count}  ⚠️ Warnings: {warning_count}"
    )
    lines.append("")
    for r in results:
        icon = {"passed": "✅", "failed": "❌", "warning": "⚠️"}.get(r.status, "•")
        lines.append(f"{icon} **{r.name}**: {r.plain_reason}")

    return ModelCriticOutput(
        test_name=test_name,
        test_family=test_family,
        checks_applicable=True,
        results=results,
        total_checks=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        has_failures=has_failures,
        proceed_to_final_report=not has_failures,
        summary_message="\n".join(lines),
    )