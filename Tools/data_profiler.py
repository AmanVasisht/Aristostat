import json
import pandas as pd
from langchain_core.tools import tool

from core.profiler_engine import profile_dataframe


# ─────────────────────────────────────────────
# IN-MEMORY DATAFRAME STORE
# Holds the currently loaded CSV so all tools
# share the same reference without re-reading the file.
# ─────────────────────────────────────────────

_dataframe_store: dict[str, pd.DataFrame] = {}


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def load_csv(filepath: str) -> str:
    """
    Load a CSV file from the given filepath into memory.
    Returns a confirmation message with basic shape info,
    or an error message if the file could not be loaded.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return "ERROR: The CSV file is empty — no rows found."
        _dataframe_store["current"] = df
        return (
            f"CSV loaded successfully. "
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns. "
            f"Columns: {list(df.columns)}"
        )
    except FileNotFoundError:
        return f"ERROR: File not found at path '{filepath}'."
    except pd.errors.EmptyDataError:
        return "ERROR: The CSV file has no data."
    except Exception as e:
        return f"ERROR: Could not load CSV — {str(e)}"


@tool
def run_profiler() -> str:
    """
    Run the full data profiling analysis on the currently loaded CSV.
    Returns the complete ProfilerOutput as a JSON string.
    Must call load_csv first.
    """
    if "current" not in _dataframe_store:
        return "ERROR: No CSV loaded. Please call load_csv first."
    try:
        result = profile_dataframe(_dataframe_store["current"])
        return result.model_dump_json(indent=2)
    except Exception as e:
        return f"ERROR: Profiling failed — {str(e)}"


@tool
def get_column_names() -> str:
    """
    Return the list of column names in the currently loaded CSV as a JSON array.
    Useful for verifying column references before profiling.
    """
    if "current" not in _dataframe_store:
        return "ERROR: No CSV loaded. Please call load_csv first."
    return json.dumps(list(_dataframe_store["current"].columns))


@tool
def get_sample_rows(n: int = 5) -> str:
    """
    Return the first n rows of the loaded CSV as a JSON string.
    Useful for a quick sanity check on data types and formatting.
    Args:
        n: Number of rows to return (default 5, max 20).
    """
    if "current" not in _dataframe_store:
        return "ERROR: No CSV loaded. Please call load_csv first."
    n = min(n, 20)
    return _dataframe_store["current"].head(n).to_json(orient="records", indent=2)


# ─────────────────────────────────────────────
# EXPORTED TOOL LIST & STORE ACCESSOR
# ─────────────────────────────────────────────

# Import this list directly in the agent file
DATA_PROFILER_TOOLS = [load_csv, run_profiler, get_column_names, get_sample_rows]


def get_dataframe_store() -> dict[str, pd.DataFrame]:
    """
    Expose the dataframe store so the agent runner can
    retrieve the loaded DataFrame for building the final profiler_output dict.
    """
    return _dataframe_store