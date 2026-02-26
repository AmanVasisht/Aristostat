"""
FILE: core/preprocessor_engine.py
-----------------------------------
Pure preprocessing logic for the Preprocessor agent.
No LangChain or LLM dependencies.

Responsibilities (universal — test-independent):
  1. Replace disguised null symbols with proper NaN
     (uses disguised_nulls_found already identified by the Data Profiler)
  2. Coerce object-dtype columns that should be numeric to float
     (uses continuous_columns list from ProfilerOutput)
  3. Handle NaN values:
       - missing < 5%  → drop affected rows (listwise deletion)
       - missing 5-20% → impute (mean/median for continuous, mode for categorical)
       - missing > 20% → fatal error, pipeline stops, user must intervene

NOT done here (test-specific, handled later):
  - Standardization / scaling
  - Categorical encoding (dummies, ordinal)
  - Log transforms or skewness corrections
  - Outlier removal
"""

import pandas as pd
import numpy as np

from Schemas.preprocessor import PreprocessorOutput, ColumnCleaningLog
from constants.preprocessor import MISSING_LOW_THRESHOLD, MISSING_MODERATE_THRESHOLD



# ─────────────────────────────────────────────
# HELPER — REPLACE DISGUISED NULLS
# ─────────────────────────────────────────────

def _replace_disguised_nulls(
    df: pd.DataFrame,
    profiler_output: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Uses disguised_nulls_found from the ProfilerOutput to replace
    known non-standard null symbols with proper NaN.
    Works on both continuous and categorical columns.

    Returns:
        df:              DataFrame with disguised nulls replaced
        replaced_counts: {column_name: count_replaced}
    """
    df = df.copy()
    replaced_counts: dict[str, int] = {}

    all_columns = (
        profiler_output.get("continuous_columns", []) +
        profiler_output.get("categorical_columns", [])
    )

    for col_info in all_columns:
        col = col_info["column"]
        symbols = col_info.get("disguised_nulls_found", [])
        if not symbols or col not in df.columns:
            continue

        count = 0
        for symbol in symbols:
            mask = df[col].astype(str).str.strip() == str(symbol).strip()
            count += int(mask.sum())
            df.loc[mask, col] = np.nan

        if count > 0:
            replaced_counts[col] = count

    return df, replaced_counts


# ─────────────────────────────────────────────
# HELPER — COERCE OBJECT COLUMNS TO NUMERIC
# ─────────────────────────────────────────────

def _coerce_numeric_columns(
    df: pd.DataFrame,
    profiler_output: dict,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    """
    Coerces columns listed as continuous in ProfilerOutput but still
    carrying object dtype (e.g. after disguised null replacement) to float.

    Returns:
        df:             DataFrame with coerced dtypes
        coerced_cols:   {column_name: (original_dtype, final_dtype)}
    """
    df = df.copy()
    coerced_cols: dict[str, tuple[str, str]] = {}

    for col_info in profiler_output.get("continuous_columns", []):
        col = col_info["column"]
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            original_dtype = str(df[col].dtype)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced_cols[col] = (original_dtype, str(df[col].dtype))

    return df, coerced_cols


# ─────────────────────────────────────────────
# HELPER — HANDLE MISSING VALUES PER COLUMN
# ─────────────────────────────────────────────

def _handle_missing_values(
    df: pd.DataFrame,
    profiler_output: dict,
) -> tuple[pd.DataFrame, dict[str, dict], list[str], list[str]]:
    """
    Applies the three-tier missing value strategy per column:
      - < 5%  → drop rows with NaN in this column
      - 5-20% → impute (mean/median for continuous, mode for categorical)
      - > 20% → fatal — collect in high_missingness_cols, do not modify

    Note: missingness is recalculated after disguised null replacement,
    since that step may have introduced new NaNs.

    Returns:
        df:                    Cleaned DataFrame
        actions:               {col: {strategy, count}} per column acted on
        high_missingness_cols: columns that exceeded 20% — fatal
        warnings:              non-fatal notices
    """
    df = df.copy()
    n_rows = len(df)
    actions: dict[str, dict] = {}
    high_missingness_cols: list[str] = []
    warnings: list[str] = []

    continuous_cols = {c["column"] for c in profiler_output.get("continuous_columns", [])}
    categorical_cols = {c["column"] for c in profiler_output.get("categorical_columns", [])}

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue

        missing_pct = null_count / n_rows

        # ── > 20% → fatal, do not touch ──
        if missing_pct > MISSING_MODERATE_THRESHOLD:
            high_missingness_cols.append(col)
            continue

        # ── < 5% → drop rows ──
        if missing_pct < MISSING_LOW_THRESHOLD:
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)
            actions[col] = {"strategy": "drop_rows", "count": dropped}

        # ── 5-20% → impute ──
        else:
            if col in continuous_cols:
                # Use median (more robust to skew than mean)
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                actions[col] = {
                    "strategy": "impute_median",
                    "count": null_count,
                    "fill_value": round(float(fill_val), 4),
                }
                warnings.append(
                    f"Column '{col}' had {null_count} missing values ({missing_pct*100:.1f}%) "
                    f"— imputed with median ({round(float(fill_val), 4)})."
                )
            elif col in categorical_cols:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                actions[col] = {
                    "strategy": "impute_mode",
                    "count": null_count,
                    "fill_value": str(fill_val),
                }
                warnings.append(
                    f"Column '{col}' had {null_count} missing values ({missing_pct*100:.1f}%) "
                    f"— imputed with mode ('{fill_val}')."
                )
            else:
                # Unknown type — drop rows as safe default
                before = len(df)
                df = df.dropna(subset=[col])
                dropped = before - len(df)
                actions[col] = {"strategy": "drop_rows_unknown_type", "count": dropped}

    return df, actions, high_missingness_cols, warnings


# ─────────────────────────────────────────────
# MAIN — PREPROCESS DATAFRAME
# ─────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    profiler_output: dict,
) -> tuple[pd.DataFrame, PreprocessorOutput]:
    """
    Main preprocessing function. Runs all three universal steps in order:
      1. Replace disguised nulls
      2. Coerce object → numeric where appropriate
      3. Handle missing values (drop / impute / fatal)

    Returns:
        cleaned_df:        The preprocessed DataFrame
        preprocessor_output: Structured summary of all changes made
    """
    original_shape = df.shape
    column_logs: dict[str, ColumnCleaningLog] = {
        col: ColumnCleaningLog(
            column=col,
            original_dtype=str(df[col].dtype),
        )
        for col in df.columns
    }
    changes_summary: list[str] = []
    warnings: list[str] = []

    # ── STEP 1: Replace disguised nulls ──
    df, replaced_counts = _replace_disguised_nulls(df, profiler_output)
    for col, count in replaced_counts.items():
        column_logs[col].disguised_nulls_replaced = count
        changes_summary.append(
            f"'{col}': replaced {count} disguised null symbol(s) with NaN."
        )

    # ── STEP 2: Coerce object → numeric ──
    df, coerced_cols = _coerce_numeric_columns(df, profiler_output)
    for col, (orig_dtype, final_dtype) in coerced_cols.items():
        column_logs[col].dtype_coerced = True
        column_logs[col].final_dtype = final_dtype
        changes_summary.append(
            f"'{col}': dtype coerced from {orig_dtype} to {final_dtype}."
        )

    # ── STEP 3: Handle missing values ──
    df, missing_actions, high_miss_cols, miss_warnings = _handle_missing_values(
        df, profiler_output
    )
    warnings.extend(miss_warnings)

    rows_dropped_total = original_shape[0] - len(df)

    for col, action in missing_actions.items():
        strategy = action["strategy"]
        count = action["count"]

        if "drop" in strategy:
            column_logs[col].rows_dropped_due_to_null = count
            changes_summary.append(
                f"'{col}': dropped {count} row(s) with missing values (<5% missing)."
            )
        elif strategy == "impute_median":
            column_logs[col].nulls_imputed = count
            column_logs[col].imputation_strategy = "median"
            # warning already added in _handle_missing_values
        elif strategy == "impute_mode":
            column_logs[col].nulls_imputed = count
            column_logs[col].imputation_strategy = "mode"
            # warning already added in _handle_missing_values

    # ── Update final dtypes in logs ──
    for col in df.columns:
        if col in column_logs:
            if column_logs[col].final_dtype is None:
                column_logs[col].final_dtype = str(df[col].dtype)

    # ── Fatal error if any column exceeded 20% missing ──
    fatal_error = None
    if high_miss_cols:
        fatal_error = (
            f"The following column(s) have more than 20% missing values and cannot "
            f"be automatically handled: {high_miss_cols}. "
            f"Please either remove these columns, provide more complete data, "
            f"or handle them manually before proceeding."
        )

    # ── Serialize cleaned DataFrame for LangGraph state ──
    cleaned_data_json = df.to_json(orient="records") if not high_miss_cols else ""

    return df, PreprocessorOutput(
        original_shape=original_shape,
        final_shape=df.shape,
        rows_dropped_total=rows_dropped_total,
        column_logs=list(column_logs.values()),
        high_missingness_fatal=high_miss_cols,
        fatal_error=fatal_error,
        changes_summary=changes_summary,
        warnings=warnings,
        cleaned_data_json=cleaned_data_json,
    )