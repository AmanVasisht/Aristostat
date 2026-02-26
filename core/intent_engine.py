"""
FILE: core/intent_engine.py
-----------------------------
Pure deterministic logic for the Intent Interpreter agent.
No LangChain or LLM dependencies — handles column validation,
dtype enrichment from ProfilerOutput, column role inference,
and top combination suggestions for open-ended queries.

Behaviour rules:
  - Invalid columns (not found in dataset) → hard fatal error, pipeline stops.
  - Ambiguous queries → best guess + low confidence flag (no clarification question).
  - Open-ended queries → suggest top column combinations from profiler output.
"""

from itertools import combinations as _combinations

from schemas.intent_schema import (
    IntentOutput,
    IntentType,
    AnalysisGoal,
    ConfidenceLevel,
    ColumnReference,
    ColumnCombination,
)


# ─────────────────────────────────────────────
# KNOWN TEST NAME ALIASES
# Maps common user phrasings to canonical test names.
# Used to detect explicit test requests reliably
# without relying solely on the LLM.
# ─────────────────────────────────────────────

KNOWN_TEST_ALIASES: dict[str, str] = {
    # Regression
    "linear regression":            "Simple Linear Regression",
    "slr":                          "Simple Linear Regression",
    "simple linear regression":     "Simple Linear Regression",
    "multiple linear regression":   "Multiple Linear Regression",
    "mlr":                          "Multiple Linear Regression",
    "multiple regression":          "Multiple Linear Regression",
    "ridge regression":             "Ridge Regression",
    "ridge":                        "Ridge Regression",
    "lasso":                        "Lasso Regression",
    "lasso regression":             "Lasso Regression",

    # ANOVA
    "anova":                        "One-Way ANOVA",
    "one way anova":                "One-Way ANOVA",
    "one-way anova":                "One-Way ANOVA",
    "two way anova":                "Two-Way ANOVA",
    "two-way anova":                "Two-Way ANOVA",

    # t-tests
    "t test":                       "Independent Samples T-Test",
    "t-test":                       "Independent Samples T-Test",
    "independent t test":           "Independent Samples T-Test",
    "independent samples t test":   "Independent Samples T-Test",
    "paired t test":                "Paired T-Test",
    "paired t-test":                "Paired T-Test",
    "one sample t test":            "One Sample T-Test",

    # Non-parametric
    "mann whitney":                 "Mann-Whitney U Test",
    "mann-whitney":                 "Mann-Whitney U Test",
    "mann whitney u":               "Mann-Whitney U Test",
    "wilcoxon":                     "Wilcoxon Signed-Rank Test",
    "kruskal wallis":               "Kruskal-Wallis Test",
    "kruskal-wallis":               "Kruskal-Wallis Test",

    # Dimensionality
    "pca":                          "Principal Component Analysis",
    "principal component analysis": "Principal Component Analysis",

    # Correlation
    "correlation":                  "Pearson Correlation",
    "pearson":                      "Pearson Correlation",
    "pearson correlation":          "Pearson Correlation",
    "spearman":                     "Spearman Correlation",
    "spearman correlation":         "Spearman Correlation",
}

# Maps canonical test names to their analysis goal
TEST_TO_GOAL: dict[str, AnalysisGoal] = {
    "Simple Linear Regression":     AnalysisGoal.RELATIONSHIP,
    "Multiple Linear Regression":   AnalysisGoal.PREDICTION,
    "Ridge Regression":             AnalysisGoal.PREDICTION,
    "Lasso Regression":             AnalysisGoal.PREDICTION,
    "One-Way ANOVA":                AnalysisGoal.INFERENCE,
    "Two-Way ANOVA":                AnalysisGoal.INFERENCE,
    "Independent Samples T-Test":   AnalysisGoal.INFERENCE,
    "Paired T-Test":                AnalysisGoal.INFERENCE,
    "One Sample T-Test":            AnalysisGoal.INFERENCE,
    "Mann-Whitney U Test":          AnalysisGoal.INFERENCE,
    "Wilcoxon Signed-Rank Test":    AnalysisGoal.INFERENCE,
    "Kruskal-Wallis Test":          AnalysisGoal.INFERENCE,
    "Principal Component Analysis": AnalysisGoal.DIMENSIONALITY,
    "Pearson Correlation":          AnalysisGoal.RELATIONSHIP,
    "Spearman Correlation":         AnalysisGoal.RELATIONSHIP,
}


# ─────────────────────────────────────────────
# HELPER — BUILD COLUMN LOOKUP FROM PROFILER OUTPUT
# ─────────────────────────────────────────────

def build_column_type_map(profiler_output: dict) -> dict[str, str]:
    """
    Build a flat map of {column_name: "continuous" | "categorical"}
    from the ProfilerOutput dict produced by the Data Profiler.
    Used to enrich ColumnReference objects with dtype info.
    """
    col_map: dict[str, str] = {}
    for col in profiler_output.get("continuous_columns", []):
        col_map[col["column"]] = "continuous"
    for col in profiler_output.get("categorical_columns", []):
        col_map[col["column"]] = "categorical"
    return col_map


# ─────────────────────────────────────────────
# HELPER — VALIDATE AND ENRICH COLUMNS
# Hard error if any mentioned column is not found.
# Case-insensitive matching is attempted first.
# ─────────────────────────────────────────────

def validate_and_enrich_columns(
    raw_columns: list[dict],       # [{name, role}] from LLM output
    col_type_map: dict[str, str],  # from build_column_type_map()
) -> tuple[list[ColumnReference], list[str], list[str]]:
    """
    Cross-references LLM-extracted column references against the actual
    dataset columns from ProfilerOutput.

    Invalid columns (not found even case-insensitively) are collected
    in invalid_cols. The caller (build_intent_output) treats any
    invalid_cols as a hard fatal error — pipeline stops and the user
    must correct the column name before proceeding.

    Returns:
        valid_cols:   List of enriched ColumnReference objects
        invalid_cols: Column names not found in dataset (triggers fatal error)
        warnings:     Soft notices (e.g. case-insensitive match was used)
    """
    valid_cols: list[ColumnReference] = []
    invalid_cols: list[str] = []
    warnings: list[str] = []

    dataset_cols_lower = {c.lower(): c for c in col_type_map.keys()}

    for raw in raw_columns:
        name = raw.get("name", "").strip()
        role = raw.get("role", "unspecified")

        # Exact match
        if name in col_type_map:
            valid_cols.append(ColumnReference(
                name=name,
                role=role,
                dtype_from_profiler=col_type_map[name]
            ))

        # Case-insensitive match (soft warning, still valid)
        elif name.lower() in dataset_cols_lower:
            actual_name = dataset_cols_lower[name.lower()]
            warnings.append(
                f"Column '{name}' matched to '{actual_name}' (case-insensitive). "
                f"Using '{actual_name}'."
            )
            valid_cols.append(ColumnReference(
                name=actual_name,
                role=role,
                dtype_from_profiler=col_type_map[actual_name]
            ))

        # Not found at all → hard error
        else:
            invalid_cols.append(name)

    return valid_cols, invalid_cols, warnings


# ─────────────────────────────────────────────
# HELPER — DETECT EXPLICIT TEST FROM QUERY
# ─────────────────────────────────────────────

def detect_explicit_test(query: str) -> str | None:
    """
    Scan the user's raw query for known test name aliases.
    Returns the canonical test name if found, else None.
    Deterministic pre-check before the LLM runs so clear cases
    like 'run a t-test' are caught reliably.
    """
    query_lower = query.lower().strip()
    for alias, canonical in KNOWN_TEST_ALIASES.items():
        if alias in query_lower:
            return canonical
    return None


# ─────────────────────────────────────────────
# HELPER — SUGGEST TOP COLUMN COMBINATIONS
# Used for open-ended queries where the user gave no direction.
# Produces meaningful pairings based on column types from the profiler.
# ─────────────────────────────────────────────

def suggest_top_combinations(profiler_output: dict) -> list[ColumnCombination]:
    """
    Generates up to 5 meaningful column combination suggestions
    for open-ended queries, based on column types from the profiler.

    Combination priority (in order):
      1. Pairs of continuous columns      → Correlation / SLR
      2. Continuous + categorical pairs   → T-Test / ANOVA
      3. 3+ continuous columns together   → MLR / PCA

    Returns a list of ColumnCombination objects (max 5).
    """
    continuous = [c["column"] for c in profiler_output.get("continuous_columns", [])]
    categorical = [c["column"] for c in profiler_output.get("categorical_columns", [])]
    suggestions: list[ColumnCombination] = []

    # Rule 1: Pairs of continuous → relationship / correlation
    for col_a, col_b in list(_combinations(continuous, 2))[:2]:
        suggestions.append(ColumnCombination(
            columns=[col_a, col_b],
            suggested_goal="relationship",
            rationale=(
                f"Both '{col_a}' and '{col_b}' are continuous — "
                f"suitable for correlation analysis or Simple Linear Regression."
            )
        ))
        if len(suggestions) >= 5:
            return suggestions

    # Rule 2: Continuous outcome + categorical grouping → inference
    for cont in continuous[:2]:
        for cat in categorical[:2]:
            suggestions.append(ColumnCombination(
                columns=[cont, cat],
                suggested_goal="inference",
                rationale=(
                    f"'{cont}' is continuous and '{cat}' is categorical — "
                    f"suitable for a T-Test or ANOVA to compare group means."
                )
            ))
            if len(suggestions) >= 5:
                return suggestions

    # Rule 3: Multiple continuous → MLR or PCA
    if len(continuous) >= 3 and len(suggestions) < 5:
        suggestions.append(ColumnCombination(
            columns=continuous[:4],
            suggested_goal="prediction",
            rationale=(
                f"Multiple continuous columns detected — "
                f"suitable for Multiple Linear Regression or PCA "
                f"for dimensionality reduction."
            )
        ))

    return suggestions[:5]


# ─────────────────────────────────────────────
# MAIN — ENRICH LLM OUTPUT INTO FINAL IntentOutput
# ─────────────────────────────────────────────

def build_intent_output(
    llm_parsed: dict,
    profiler_output: dict,
    original_query: str,
) -> IntentOutput:
    """
    Takes the raw structured dict produced by the LLM tool call
    and enriches it into a final validated IntentOutput.

    Raises ValueError if any mentioned column does not exist in the dataset.
    This is a hard stop — the tool surfaces it to the user who must correct
    the column name before the pipeline can continue.

    llm_parsed keys expected:
        intent_type, analysis_goal, confidence,
        requested_test (optional), columns (list of {name, role}),
        interpretation_summary
    """
    col_type_map = build_column_type_map(profiler_output)

    # ── Deterministic test detection override ──
    # If the raw query clearly names a test, trust that over the LLM.
    detected_test = detect_explicit_test(original_query)
    requested_test = detected_test or llm_parsed.get("requested_test")

    # ── Methodologist bypass if explicit test was named ──
    methodologist_bypass = (
        requested_test is not None
        and llm_parsed.get("intent_type") == IntentType.EXPLICIT_TEST
    )

    # ── Resolve analysis goal ──
    analysis_goal_str = llm_parsed.get("analysis_goal", AnalysisGoal.UNKNOWN)
    if requested_test and requested_test in TEST_TO_GOAL:
        analysis_goal = TEST_TO_GOAL[requested_test]
    else:
        try:
            analysis_goal = AnalysisGoal(analysis_goal_str)
        except ValueError:
            analysis_goal = AnalysisGoal.UNKNOWN

    # ── Parse enums safely ──
    try:
        intent_type = IntentType(llm_parsed.get("intent_type", IntentType.OPEN_ENDED))
    except ValueError:
        intent_type = IntentType.OPEN_ENDED

    try:
        confidence = ConfidenceLevel(llm_parsed.get("confidence", ConfidenceLevel.MEDIUM))
    except ValueError:
        confidence = ConfidenceLevel.MEDIUM

    # ── Column validation — HARD ERROR if any column not found ──
    raw_columns = llm_parsed.get("columns", [])
    valid_cols, invalid_cols, col_warnings = validate_and_enrich_columns(
        raw_columns, col_type_map
    )

    if invalid_cols:
        # Raise immediately — parse_intent_from_llm tool catches this
        # and surfaces a clear error message to the user.
        raise ValueError(
            f"The following column(s) were not found in the dataset: {invalid_cols}. "
            f"Available columns are: {list(col_type_map.keys())}. "
            f"Please correct the column name(s) and try again."
        )

    # ── Open-ended: generate top combination suggestions ──
    suggested_combinations: list[ColumnCombination] = []
    if intent_type == IntentType.OPEN_ENDED:
        suggested_combinations = suggest_top_combinations(profiler_output)

    return IntentOutput(
        intent_type=intent_type,
        analysis_goal=analysis_goal,
        confidence=confidence,
        requested_test=requested_test,
        methodologist_bypass=methodologist_bypass,
        columns=valid_cols,
        all_columns_mode=(intent_type == IntentType.OPEN_ENDED),
        invalid_columns=[],          # always empty here — raised above if any existed
        column_warnings=col_warnings,
        suggested_combinations=suggested_combinations,
        interpretation_summary=llm_parsed.get("interpretation_summary", ""),
        clarification_needed=False,  # never ask — ambiguous = best guess + low confidence
        clarification_question=None,
        original_query=original_query,
    )