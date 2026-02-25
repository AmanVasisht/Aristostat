"""
FILE: schemas/data_profiler_schema.py
--------------------------------------
Pydantic output schemas for the Data Profiler agent.
These are shared data contracts — downstream agents (Intent Interpreter,
Methodologist, etc.) will also import ProfilerOutput from here.
"""

from pydantic import BaseModel, Field


class ContinuousColumnProfile(BaseModel):
    column: str
    dtype: str
    missing_count: int
    missing_pct: float
    missing_severity: str                        # "low" | "moderate" | "high"
    disguised_nulls_found: list[str] = Field(default_factory=list)
    disguised_null_count: int = 0
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    variance: float | None = None
    min: float | None = None
    max: float | None = None
    range: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    skewness_interpretation: str | None = None  # "symmetric" | "moderate skew" | "high skew"
    kurtosis: float | None = None
    confidence_interval_95: tuple[float, float] | None = None
    anomaly_count: int = 0
    anomaly_pct: float = 0.0
    anomaly_indices: list[int] = Field(default_factory=list)


class CategoricalColumnProfile(BaseModel):
    column: str
    dtype: str
    missing_count: int
    missing_pct: float
    missing_severity: str
    disguised_nulls_found: list[str] = Field(default_factory=list)
    disguised_null_count: int = 0
    unique_values: list[str]
    cardinality: int
    high_cardinality: bool
    mode: str | None = None
    mode_frequency: float | None = None         # proportion of mode in column
    value_counts: dict[str, int] = Field(default_factory=dict)
    class_imbalance_flag: bool = False           # True if dominant class > 80%


class ProfilerOutput(BaseModel):
    n_rows: int
    n_cols: int
    total_missing_cells: int
    total_missing_pct: float
    continuous_columns: list[ContinuousColumnProfile] = Field(default_factory=list)
    categorical_columns: list[CategoricalColumnProfile] = Field(default_factory=list)
    unresolvable_columns: list[str] = Field(default_factory=list)  # all-empty or unparseable
    warnings: list[str] = Field(default_factory=list)
    fatal_errors: list[str] = Field(default_factory=list)          # genuine blockers