# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Common symbols that represent missing values but aren't NaN
DISGUISED_NULL_SYMBOLS = {
    "-", "--", "---", "n/a", "na", "N/A", "NA", "nil", "Nil", "NIL",
    "none", "None", "NONE", "null", "Null", "NULL", "?", "*", "missing",
    "Missing", "MISSING", "unknown", "Unknown", "UNKNOWN", ".", " ", ""
}

# Missingness severity thresholds
MISSING_LOW_THRESHOLD = 0.05       # < 5%  → low, manageable
MISSING_MODERATE_THRESHOLD = 0.20  # < 20% → moderate, worth noting
# >= 20% → serious concern

# Anomaly detection: IQR multiplier
IQR_MULTIPLIER = 1.5

# Skewness thresholds
SKEW_MODERATE = 0.5
SKEW_HIGH = 1.0

# High cardinality threshold for categorical columns
HIGH_CARDINALITY_THRESHOLD = 50