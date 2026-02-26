"""
FILE: configs/post_test_assumptions.py
----------------------------------------
Registry mapping test families to their post-test assumption checks.
These checks require a fitted model and cannot be run before the test.

Keyed by test_family (from TestFamily enum in statistician_schema.py):
  "regression"     → SLR, MLR, Ridge, Lasso
  "inference"      → skipped (pass straight to Final Report)
  "correlation"    → skipped (pass straight to Final Report)
  "dimensionality" → skipped (pass straight to Final Report)

Each check entry matches the same structure as configs/assumptions.py:
  name, description, check_method, test_fn, alpha
"""

POST_TEST_ASSUMPTION_REGISTRY: dict[str, list[dict]] = {

    # ─────────────────────────────────────────────
    # REGRESSION (SLR, MLR, Ridge, Lasso)
    # All regression models produce residuals
    # ─────────────────────────────────────────────
    "regression": [
        {
            "name": "normality_of_residuals",
            "description": "Residuals should be approximately normally distributed.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_of_residuals",
            "alpha": 0.05,
        },
        {
            "name": "homoscedasticity",
            "description": "Variance of residuals should be constant across fitted values (no heteroscedasticity).",
            "check_method": "statistical_test",
            "test_fn": "check_homoscedasticity_bp",
            "alpha": 0.05,
        },
        {
            "name": "no_autocorrelation",
            "description": "Residuals should not be correlated with each other (Durbin-Watson statistic should be close to 2).",
            "check_method": "statistical_test",
            "test_fn": "check_autocorrelation_dw",
            "alpha": None,   # DW uses range check, not p-value
        },
        {
            "name": "no_influential_points",
            "description": "No single observation should have disproportionate influence on the model (Cook's distance < 4/n).",
            "check_method": "heuristic",
            "test_fn": "check_influential_points_cooks",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # INFERENCE, CORRELATION, DIMENSIONALITY
    # No post-test residual checks needed — pass straight to Final Report
    # ─────────────────────────────────────────────
    "inference":      [],
    "correlation":    [],
    "dimensionality": [],
}