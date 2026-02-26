"""
FILE: schemas/statistician_schema.py
--------------------------------------
Pydantic output schemas for the Statistician agent.
Separate result schemas per test type family — each captures exactly
the outputs that test produces, no more.

Test type families:
  - InferenceResult      : T-Tests, ANOVA, Mann-Whitney, Wilcoxon, Kruskal-Wallis
  - RegressionResult     : SLR, MLR, Ridge, Lasso
  - CorrelationResult    : Pearson, Spearman
  - DimensionalityResult : PCA

StatisticianOutput wraps whichever result type was produced.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class TestFamily(str, Enum):
    INFERENCE      = "inference"
    REGRESSION     = "regression"
    CORRELATION    = "correlation"
    DIMENSIONALITY = "dimensionality"


class SignificanceVerdict(str, Enum):
    SIGNIFICANT     = "significant"
    NOT_SIGNIFICANT = "not_significant"
    BORDERLINE      = "borderline"    # p within 10% of alpha


# ─────────────────────────────────────────────
# REGRESSION COEFFICIENT
# ─────────────────────────────────────────────

class Coefficient(BaseModel):
    variable:   str
    estimate:   float
    std_error:  float | None = None
    t_statistic: float | None = None
    p_value:    float | None = None
    ci_lower:   float | None = None    # 95% confidence interval lower bound
    ci_upper:   float | None = None    # 95% confidence interval upper bound


# ─────────────────────────────────────────────
# PCA COMPONENT
# ─────────────────────────────────────────────

class PCAComponent(BaseModel):
    component_number:       int
    explained_variance:     float       # absolute variance explained
    explained_variance_pct: float       # percentage of total variance explained
    cumulative_variance_pct: float      # cumulative % up to this component
    loadings: dict[str, float] = Field(default_factory=dict)  # {variable: loading}


# ─────────────────────────────────────────────
# RESULT SCHEMAS — ONE PER TEST FAMILY
# ─────────────────────────────────────────────

class InferenceResult(BaseModel):
    """
    For: Independent T-Test, Paired T-Test, One Sample T-Test,
         One-Way ANOVA, Two-Way ANOVA,
         Mann-Whitney U, Wilcoxon, Kruskal-Wallis
    """
    test_name:   str
    statistic:   float                  # t, F, U, W, or H depending on test
    statistic_label: str                # "t", "F", "U", "W", "H"
    p_value:     float
    alpha:       float = 0.05
    verdict:     SignificanceVerdict

    # ── Group descriptives ──
    group_stats: dict[str, dict] = Field(default_factory=dict)
    # e.g. {"M": {"mean": 55000, "std": 12000, "n": 110}, "F": {...}}

    # ── Degrees of freedom ──
    df:          float | None = None    # degrees of freedom (or tuple for F-test)
    df_between:  float | None = None    # for ANOVA
    df_within:   float | None = None    # for ANOVA

    # ── Effect size ──
    effect_size:       float | None = None
    effect_size_label: str | None = None    # "Cohen's d", "eta-squared", "r"

    # ── Correction applied ──
    correction_applied: str | None = None   # e.g. "welch"

    # ── Plain English interpretation ──
    interpretation: str = ""


class RegressionResult(BaseModel):
    """
    For: Simple Linear Regression, Multiple Linear Regression,
         Ridge Regression, Lasso Regression
    """
    test_name:   str
    coefficients: list[Coefficient] = Field(default_factory=list)

    # ── Model fit ──
    r_squared:         float | None = None
    adj_r_squared:     float | None = None
    f_statistic:       float | None = None
    f_p_value:         float | None = None
    aic:               float | None = None
    bic:               float | None = None
    rmse:              float | None = None

    # ── For Ridge/Lasso ──
    alpha_regularization: float | None = None   # regularisation parameter (not significance alpha)
    selected_features:    list[str] = Field(default_factory=list)   # Lasso non-zero features

    # ── Correction applied ──
    correction_applied: str | None = None   # e.g. "robust_se", "newey_west"

    # ── Plain English interpretation ──
    interpretation: str = ""


class CorrelationResult(BaseModel):
    """For: Pearson Correlation, Spearman Correlation"""
    test_name:   str
    statistic:   float          # r or rho
    statistic_label: str        # "r" or "rho"
    p_value:     float
    alpha:       float = 0.05
    verdict:     SignificanceVerdict

    # ── Confidence interval (Pearson only via Fisher Z) ──
    ci_lower:    float | None = None
    ci_upper:    float | None = None

    # ── Strength interpretation ──
    correlation_strength: str = ""   # "weak", "moderate", "strong"

    # ── Plain English interpretation ──
    interpretation: str = ""


class DimensionalityResult(BaseModel):
    """For: Principal Component Analysis"""
    test_name:   str = "Principal Component Analysis"
    n_components_total: int
    n_components_selected: int      # components needed to explain >= 80% variance

    components: list[PCAComponent] = Field(default_factory=list)

    # ── Variance summary ──
    variance_explained_80pct: int   # number of components to reach 80% explained variance
    total_variance_explained: float # by selected components

    # ── Plain English interpretation ──
    interpretation: str = ""


# ─────────────────────────────────────────────
# MAIN OUTPUT SCHEMA
# ─────────────────────────────────────────────

class StatisticianOutput(BaseModel):
    test_name:   str
    test_family: TestFamily

    # ── Exactly one of these will be populated ──
    inference_result:      InferenceResult      | None = None
    regression_result:     RegressionResult     | None = None
    correlation_result:    CorrelationResult    | None = None
    dimensionality_result: DimensionalityResult | None = None

    # ── Metadata ──
    n_observations:     int | None = None
    correction_applied: str | None = None
    columns_used: dict[str, str] = Field(default_factory=dict)
    # e.g. {"dependent": "salary", "grouping": "gender"}

    # ── Flag for Model Critic ──
    model_available_in_memory: bool = False
    # True if a fitted statsmodels/sklearn model is stored in the tools session store

    # ── Warnings carried forward ──
    warnings: list[str] = Field(default_factory=list)