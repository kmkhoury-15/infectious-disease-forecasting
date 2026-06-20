"""File loading and column inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}

DATE_CANDIDATES = ["date", "event_date", "dtevent", "week", "week_end", "sample_date", "report_date"]
VALUE_CANDIDATES = ["cases", "case_count", "weekly_case_count", "count", "n", "value"]
DISEASE_CANDIDATES = ["disease", "condition", "morbspc", "pathogen", "diagnosis"]
GEOGRAPHY_CANDIDATES = ["county", "geography", "jurisdiction", "region", "state", "zip"]


def load_case_data(file_path: str | Path) -> pd.DataFrame:
    """Load case data from CSV, TXT/TSV, XLSX, or XLS files."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, low_memory=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped string column names."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def infer_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Infer a likely column name using exact and contains-based matching."""
    columns_list = list(columns)
    lowered = {str(col).strip().lower(): col for col in columns_list}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for col in columns_list:
        col_lower = str(col).lower()
        if any(candidate in col_lower for candidate in candidates):
            return col

    return None


def infer_schema(df: pd.DataFrame) -> dict[str, str | None]:
    """Infer date, value, disease, and geography columns from a dataframe."""
    return {
        "date_col": infer_column(df.columns, DATE_CANDIDATES),
        "value_col": infer_column(df.columns, VALUE_CANDIDATES),
        "disease_col": infer_column(df.columns, DISEASE_CANDIDATES),
        "geography_col": infer_column(df.columns, GEOGRAPHY_CANDIDATES),
    }
