"""
FILE: configs/assumptions.py
------------------------------
Central registry mapping each supported statistical test to its
required pre-test assumptions.

Each assumption entry has:
  name:         Short identifier used in code
  description:  Plain English description shown to user
  check_method: How it is verified:
                  "statistical_test" — run a specific test (Shapiro, Levene, etc.)
                  "heuristic"        — checked via data properties (sample size, dtype, etc.)
                  "manual"           — cannot be verified programmatically, user must confirm
  test_fn:      Name of the function in assumption_engine.py that runs the check
                (None for manual checks)
  alpha:        Significance level for statistical checks (default 0.05)
"""

ASSUMPTION_REGISTRY: dict[str, list[dict]] = {

    # ─────────────────────────────────────────────
    # SIMPLE LINEAR REGRESSION
    # ─────────────────────────────────────────────
    "Simple Linear Regression": [
        {
            "name": "linearity",
            "description": "The relationship between the independent and dependent variable should be linear.",
            "check_method": "heuristic",
            "test_fn": "check_linearity_heuristic",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent of each other (no autocorrelation in residuals).",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "no_significant_outliers",
            "description": "No extreme outliers should unduly influence the regression line.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # MULTIPLE LINEAR REGRESSION
    # ─────────────────────────────────────────────
    "Multiple Linear Regression": [
        {
            "name": "linearity",
            "description": "Each independent variable should have a linear relationship with the dependent variable.",
            "check_method": "heuristic",
            "test_fn": "check_linearity_heuristic",
            "alpha": None,
        },
        {
            "name": "no_multicollinearity",
            "description": "Independent variables should not be highly correlated with each other (VIF < 10).",
            "check_method": "heuristic",
            "test_fn": "check_multicollinearity_vif",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent of each other.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "no_significant_outliers",
            "description": "No extreme outliers should unduly influence the model.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # RIDGE / LASSO REGRESSION
    # ─────────────────────────────────────────────
    "Ridge Regression": [
        {
            "name": "no_multicollinearity",
            "description": "Ridge is used precisely when multicollinearity exists — VIF check is informational here.",
            "check_method": "heuristic",
            "test_fn": "check_multicollinearity_vif",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent of each other.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Sample size should be adequate relative to number of predictors (n > 10 × p).",
            "check_method": "heuristic",
            "test_fn": "check_sample_size",
            "alpha": None,
        },
    ],
    "Lasso Regression": [
        {
            "name": "no_multicollinearity",
            "description": "Lasso handles multicollinearity via feature selection — VIF check is informational.",
            "check_method": "heuristic",
            "test_fn": "check_multicollinearity_vif",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent of each other.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Sample size should be adequate relative to number of predictors.",
            "check_method": "heuristic",
            "test_fn": "check_sample_size",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # ONE-WAY ANOVA
    # ─────────────────────────────────────────────
    "One-Way ANOVA": [
        {
            "name": "normality",
            "description": "The dependent variable should be approximately normally distributed within each group.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_shapiro_by_group",
            "alpha": 0.05,
        },
        {
            "name": "homogeneity_of_variance",
            "description": "Variance of the dependent variable should be equal across all groups (homoscedasticity).",
            "check_method": "statistical_test",
            "test_fn": "check_homogeneity_levene",
            "alpha": 0.05,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent within and across groups.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Each group should have at least 5 observations.",
            "check_method": "heuristic",
            "test_fn": "check_group_sample_sizes",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # TWO-WAY ANOVA
    # ─────────────────────────────────────────────
    "Two-Way ANOVA": [
        {
            "name": "normality",
            "description": "The dependent variable should be approximately normally distributed within each group combination.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_shapiro_by_group",
            "alpha": 0.05,
        },
        {
            "name": "homogeneity_of_variance",
            "description": "Variance should be equal across all group combinations.",
            "check_method": "statistical_test",
            "test_fn": "check_homogeneity_levene",
            "alpha": 0.05,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Each group combination should have sufficient observations.",
            "check_method": "heuristic",
            "test_fn": "check_group_sample_sizes",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # INDEPENDENT SAMPLES T-TEST
    # ─────────────────────────────────────────────
    "Independent Samples T-Test": [
        {
            "name": "normality",
            "description": "The dependent variable should be approximately normally distributed in each group.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_shapiro_by_group",
            "alpha": 0.05,
        },
        {
            "name": "homogeneity_of_variance",
            "description": "Variance should be approximately equal between the two groups (Levene's test).",
            "check_method": "statistical_test",
            "test_fn": "check_homogeneity_levene",
            "alpha": 0.05,
        },
        {
            "name": "independence_of_observations",
            "description": "The two groups must be independent of each other.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Each group should have at least 5 observations.",
            "check_method": "heuristic",
            "test_fn": "check_group_sample_sizes",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # PAIRED T-TEST
    # ─────────────────────────────────────────────
    "Paired T-Test": [
        {
            "name": "normality_of_differences",
            "description": "The differences between paired observations should be approximately normally distributed.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_of_differences",
            "alpha": 0.05,
        },
        {
            "name": "paired_observations",
            "description": "Each observation in one group must have a corresponding observation in the other group.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "no_significant_outliers",
            "description": "No extreme outliers in the differences between pairs.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # ONE SAMPLE T-TEST
    # ─────────────────────────────────────────────
    "One Sample T-Test": [
        {
            "name": "normality",
            "description": "The variable should be approximately normally distributed.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_shapiro",
            "alpha": 0.05,
        },
        {
            "name": "no_significant_outliers",
            "description": "No extreme outliers that could distort the mean.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # MANN-WHITNEY U TEST
    # ─────────────────────────────────────────────
    "Mann-Whitney U Test": [
        {
            "name": "ordinal_or_continuous_dependent",
            "description": "The dependent variable should be at least ordinal (continuous or ranked).",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "The two groups must be independent of each other.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "similar_distribution_shape",
            "description": "For interpreting as median comparison, both groups should have similarly shaped distributions.",
            "check_method": "heuristic",
            "test_fn": "check_distribution_shape_similarity",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # WILCOXON SIGNED-RANK TEST
    # ─────────────────────────────────────────────
    "Wilcoxon Signed-Rank Test": [
        {
            "name": "paired_observations",
            "description": "Data must consist of matched pairs.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "ordinal_or_continuous_dependent",
            "description": "The dependent variable should be at least ordinal.",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
        {
            "name": "symmetry_of_differences",
            "description": "The distribution of differences between pairs should be symmetric.",
            "check_method": "heuristic",
            "test_fn": "check_symmetry_of_differences",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # KRUSKAL-WALLIS TEST
    # ─────────────────────────────────────────────
    "Kruskal-Wallis Test": [
        {
            "name": "ordinal_or_continuous_dependent",
            "description": "The dependent variable should be at least ordinal.",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent within and across groups.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
        {
            "name": "sufficient_sample_size",
            "description": "Each group should have at least 5 observations.",
            "check_method": "heuristic",
            "test_fn": "check_group_sample_sizes",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # PRINCIPAL COMPONENT ANALYSIS
    # ─────────────────────────────────────────────
    "Principal Component Analysis": [
        {
            "name": "sufficient_sample_size",
            "description": "Sample size should be at least 5× the number of variables (ideally 10× or more).",
            "check_method": "heuristic",
            "test_fn": "check_sample_size",
            "alpha": None,
        },
        {
            "name": "continuous_variables",
            "description": "All variables used in PCA should be continuous.",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
        {
            "name": "correlation_among_variables",
            "description": "Variables should show some correlation with each other (PCA is not meaningful on uncorrelated data).",
            "check_method": "heuristic",
            "test_fn": "check_correlation_matrix",
            "alpha": None,
        },
        {
            "name": "no_significant_outliers",
            "description": "Extreme outliers can disproportionately influence principal components.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # PEARSON CORRELATION
    # ─────────────────────────────────────────────
    "Pearson Correlation": [
        {
            "name": "normality",
            "description": "Both variables should be approximately normally distributed.",
            "check_method": "statistical_test",
            "test_fn": "check_normality_shapiro",
            "alpha": 0.05,
        },
        {
            "name": "linearity",
            "description": "The relationship between the two variables should be linear.",
            "check_method": "heuristic",
            "test_fn": "check_linearity_heuristic",
            "alpha": None,
        },
        {
            "name": "no_significant_outliers",
            "description": "Outliers can heavily distort the Pearson correlation coefficient.",
            "check_method": "heuristic",
            "test_fn": "check_outliers_iqr",
            "alpha": None,
        },
        {
            "name": "continuous_variables",
            "description": "Both variables should be continuous.",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
    ],

    # ─────────────────────────────────────────────
    # SPEARMAN CORRELATION
    # ─────────────────────────────────────────────
    "Spearman Correlation": [
        {
            "name": "ordinal_or_continuous",
            "description": "Both variables should be at least ordinal.",
            "check_method": "heuristic",
            "test_fn": "check_dtype_continuous",
            "alpha": None,
        },
        {
            "name": "monotonic_relationship",
            "description": "The relationship between variables should be monotonic (consistently increasing or decreasing).",
            "check_method": "heuristic",
            "test_fn": "check_monotonic_relationship",
            "alpha": None,
        },
        {
            "name": "independence_of_observations",
            "description": "Observations should be independent.",
            "check_method": "manual",
            "test_fn": None,
            "alpha": None,
        },
    ],
}