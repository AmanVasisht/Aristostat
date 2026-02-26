
# ─────────────────────────────────────────────
# THRESHOLDS (mirrors Data Profiler constants)
# ─────────────────────────────────────────────

MISSING_LOW_THRESHOLD      = 0.05   # < 5%  → drop rows
MISSING_MODERATE_THRESHOLD = 0.20   # < 20% → impute
# >= 20% → fatal error