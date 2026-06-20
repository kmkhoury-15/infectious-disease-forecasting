"""General utility helpers for the infectious disease forecasting project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np


SUPPORTED_FREQUENCIES = {
    "daily": "D",
    "weekly": "W-SUN",
    "quarterly": "Q",
    "yearly": "Y",
}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(value: str, default: str = "forecast") -> str:
    """Return a filesystem-safe name for downloaded outputs."""
    value = (value or default).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or default


def get_frequency_alias(frequency: str) -> str:
    """Map a user-friendly frequency name to a pandas frequency alias."""
    key = str(frequency).strip().lower()
    if key not in SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"Unsupported frequency '{frequency}'. Choose one of: "
            f"{', '.join(SUPPORTED_FREQUENCIES)}."
        )
    return SUPPORTED_FREQUENCIES[key]


def get_aggregation_function(aggregation: str) -> Callable:
    """Return a numpy aggregation function."""
    key = str(aggregation).strip().lower()
    if key == "sum":
        return np.sum
    if key in {"mean", "average", "avg"}:
        return np.mean
    raise ValueError("aggregation must be either 'sum' or 'mean'.")


def validate_positive_int(value: int | str, name: str, minimum: int = 1) -> int:
    """Validate and coerce a positive integer setting."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed
