"""
FILE: core/profiler_engine.py
------------------------------
Pure statistical engine for the Data Profiler agent.
No LangChain dependencies — just pandas, scipy, and numpy.
Can be unit tested independently without spinning up any LLM.
"""

import pandas as pd
import numpy as np
from scipy import stats

from Schemas.data_profiler_schema import (
    ContinuousColumnProfile,
    CategoricalColumnProfile,
    ProfilerOutput,
)
from constants.data_profiler_constants import DISGUISED_NULL_SYMBOLS, MISSING_LOW_THRESHOLD, MISSING_MODERATE_THRESHOLD, IQR_MULTIPLIER, SKEW_MODERATE, SKEW_HIGH, HIGH_CARDINALITY_THRESHOLD




# ─────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────

def _missing_severity(pct: float) -> str:
    if pct < MISSING_LOW_THRESHOLD:
        return "low"
    elif pct < MISSING_MODERATE_THRESHOLD:
        return "moderate"
    else:
        return "high"


def _skewness_interpretation(skew: float) -> str:
    abs_skew = abs(skew)
    if abs_skew < SKEW_MODERATE:
        return "symmetric"
    elif abs_skew < SKEW_HIGH:
        return "moderate skew"
    else:
        direction = "right" if skew > 0 else "left"
        return f"high {direction} skew"


def _detect_disguised_nulls(series: pd.Series) -> tuple[list[str], int]:
    """
    For object-dtype columns: find values that match known null symbols.
    Returns (list of unique disguised null symbols found, total count of such cells).
    """
    found_symbols = []
    total_count = 0
    for val in series.dropna().unique():
        str_val = str(val).strip()
        if str_val in DISGUISED_NULL_SYMBOLS or str_val.lower() in DISGUISED_NULL_SYMBOLS:
            found_symbols.append(str_val)
            total_count += int((series == val).sum())
    return found_symbols, total_count


def _detect_anomalies_iqr(series: pd.Series) -> tuple[int, float, list[int]]:
    """
    IQR-based anomaly detection on a numeric series.
    Returns (count, percentage, list of integer positional indices).
    """
    clean = series.dropna()
    if len(clean) < 4:
        return 0, 0.0, []
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    mask = (series < lower) | (series > upper)
    indices = list(series[mask].index.tolist())
    count = int(mask.sum())
    pct = round(count / len(series), 4)
    return count, pct, indices


def _confidence_interval_95(series: pd.Series) -> tuple[float, float] | None:
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        return None
    se = stats.sem(clean)
    ci = stats.t.interval(0.95, df=n - 1, loc=clean.mean(), scale=se)
    return (round(ci[0], 4), round(ci[1], 4))


def _try_coerce_to_numeric(series: pd.Series) -> pd.Series | None:
    """
    Attempt to coerce an object-dtype column to numeric.
    Returns the coerced series if >= 70% of non-null values convert successfully,
    else None.
    """
    coerced = pd.to_numeric(series, errors="coerce")
    original_non_null = series.notna().sum()
    if original_non_null == 0:
        return None
    success_rate = coerced.notna().sum() / original_non_null
    if success_rate >= 0.70:
        return coerced
    return None


# ─────────────────────────────────────────────
# PUBLIC — MAIN PROFILING FUNCTION
# ─────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> ProfilerOutput:
    """
    Main profiling function. Iterates over all columns, classifies each as
    continuous or categorical, computes stats, detects anomalies and disguised nulls.
    Pure Python — no LLM calls, no side effects, no mutations to df.
    """
    n_rows, n_cols = df.shape
    total_missing_cells = int(df.isna().sum().sum())
    total_missing_pct = round(total_missing_cells / (n_rows * n_cols), 4)

    continuous_profiles: list[ContinuousColumnProfile] = []
    categorical_profiles: list[CategoricalColumnProfile] = []
    unresolvable: list[str] = []
    warnings: list[str] = []
    fatal_errors: list[str] = []

    for col in df.columns:
        series = df[col]

        # ── All values missing → unresolvable ──
        if series.isna().all():
            unresolvable.append(col)
            warnings.append(f"Column '{col}' is entirely empty and cannot be profiled.")
            continue

        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / n_rows, 4)
        severity = _missing_severity(missing_pct)

        # ── Disguised null detection (always run on object columns) ──
        disguised_symbols: list[str] = []
        disguised_count: int = 0
        if series.dtype == object:
            disguised_symbols, disguised_count = _detect_disguised_nulls(series)
            if disguised_symbols:
                warnings.append(
                    f"Column '{col}' contains non-standard missing value symbols "
                    f"{disguised_symbols} in {disguised_count} row(s). "
                    f"These will need to be replaced with NaN during preprocessing."
                )

        # ── Type resolution ──
        # Case 1: Already numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            resolved_series = series
            is_continuous = True

        # Case 2: Object dtype — try coercing to numeric
        elif series.dtype == object:
            coerced = _try_coerce_to_numeric(series)
            if coerced is not None:
                resolved_series = coerced
                is_continuous = True
                if disguised_count == 0:
                    new_nulls = int(coerced.isna().sum()) - missing_count
                    if new_nulls > 0:
                        warnings.append(
                            f"Column '{col}' appears numeric but {new_nulls} value(s) "
                            f"could not be parsed and will be treated as missing."
                        )
            else:
                resolved_series = series
                is_continuous = False

        # Case 3: Boolean or datetime → treat as categorical
        else:
            resolved_series = series.astype(str)
            is_continuous = False

        # ── Missing severity warnings ──
        if severity == "moderate":
            warnings.append(
                f"Column '{col}' has {missing_pct*100:.1f}% missing values — moderate, review before analysis."
            )
        elif severity == "high":
            warnings.append(
                f"Column '{col}' has {missing_pct*100:.1f}% missing values — serious concern, may affect analysis."
            )

        # ─────────────────────────────────────────
        # CONTINUOUS COLUMN PROFILING
        # ─────────────────────────────────────────
        if is_continuous:
            clean = resolved_series.dropna()

            mean_val    = round(float(clean.mean()), 4)    if len(clean) > 0 else None
            median_val  = round(float(clean.median()), 4)  if len(clean) > 0 else None
            std_val     = round(float(clean.std()), 4)     if len(clean) > 1 else None
            var_val     = round(float(clean.var()), 4)     if len(clean) > 1 else None
            min_val     = round(float(clean.min()), 4)     if len(clean) > 0 else None
            max_val     = round(float(clean.max()), 4)     if len(clean) > 0 else None
            range_val   = round(max_val - min_val, 4)      if (min_val is not None and max_val is not None) else None
            q1_val      = round(float(clean.quantile(0.25)), 4) if len(clean) > 0 else None
            q3_val      = round(float(clean.quantile(0.75)), 4) if len(clean) > 0 else None
            iqr_val     = round(q3_val - q1_val, 4)        if (q1_val is not None and q3_val is not None) else None
            skew_val    = round(float(clean.skew()), 4)    if len(clean) > 2 else None
            skew_interp = _skewness_interpretation(skew_val) if skew_val is not None else None
            kurt_val    = round(float(clean.kurtosis()), 4) if len(clean) > 3 else None
            ci          = _confidence_interval_95(resolved_series)

            anomaly_count, anomaly_pct, anomaly_indices = _detect_anomalies_iqr(resolved_series)

            if skew_val is not None and abs(skew_val) >= SKEW_HIGH:
                warnings.append(
                    f"Column '{col}' shows {skew_interp} (skewness={skew_val}). "
                    f"May require transformation depending on the test selected."
                )

            if anomaly_count > 0:
                warnings.append(
                    f"Column '{col}' has {anomaly_count} anomalies detected via IQR "
                    f"({anomaly_pct*100:.1f}% of rows)."
                )

            continuous_profiles.append(ContinuousColumnProfile(
                column=col,
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_pct=missing_pct,
                missing_severity=severity,
                disguised_nulls_found=disguised_symbols,
                disguised_null_count=disguised_count,
                mean=mean_val,
                median=median_val,
                std=std_val,
                variance=var_val,
                min=min_val,
                max=max_val,
                range=range_val,
                q1=q1_val,
                q3=q3_val,
                iqr=iqr_val,
                skewness=skew_val,
                skewness_interpretation=skew_interp,
                kurtosis=kurt_val,
                confidence_interval_95=ci,
                anomaly_count=anomaly_count,
                anomaly_pct=anomaly_pct,
                anomaly_indices=anomaly_indices[:50]    # cap to avoid huge payloads
            ))

        # ─────────────────────────────────────────
        # CATEGORICAL COLUMN PROFILING
        # ─────────────────────────────────────────
        else:
            str_series = resolved_series.astype(str)
            exclude = set(str(s) for s in disguised_symbols)
            filtered = str_series[~str_series.isin(exclude) & str_series.notna()]

            unique_vals = [str(v) for v in filtered.unique().tolist()]
            cardinality = len(unique_vals)
            high_card   = cardinality > HIGH_CARDINALITY_THRESHOLD

            vc = filtered.value_counts()
            mode_val  = str(vc.index[0])             if len(vc) > 0 else None
            mode_freq = round(vc.iloc[0] / len(filtered), 4) if len(filtered) > 0 else None
            value_counts_dict = {str(k): int(v) for k, v in vc.items()}

            class_imbalance = (mode_freq is not None and mode_freq > 0.80)

            if high_card:
                warnings.append(
                    f"Column '{col}' has very high cardinality ({cardinality} unique values). "
                    f"It may be an ID or free-text column — verify before using in analysis."
                )

            if class_imbalance:
                warnings.append(
                    f"Column '{col}' has class imbalance: '{mode_val}' accounts for "
                    f"{mode_freq*100:.1f}% of values."
                )

            categorical_profiles.append(CategoricalColumnProfile(
                column=col,
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_pct=missing_pct,
                missing_severity=severity,
                disguised_nulls_found=disguised_symbols,
                disguised_null_count=disguised_count,
                unique_values=unique_vals[:100],        # cap for large categoricals
                cardinality=cardinality,
                high_cardinality=high_card,
                mode=mode_val,
                mode_frequency=mode_freq,
                value_counts=dict(list(value_counts_dict.items())[:50]),
                class_imbalance_flag=class_imbalance
            ))

    # ── Fatal error: nothing could be profiled ──
    if not continuous_profiles and not categorical_profiles:
        fatal_errors.append(
            "No columns could be profiled. The dataset may be empty or entirely unparseable."
        )

    return ProfilerOutput(
        n_rows=n_rows,
        n_cols=n_cols,
        total_missing_cells=total_missing_cells,
        total_missing_pct=total_missing_pct,
        continuous_columns=continuous_profiles,
        categorical_columns=categorical_profiles,
        unresolvable_columns=unresolvable,
        warnings=warnings,
        fatal_errors=fatal_errors
    )