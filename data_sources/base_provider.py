"""Base classes for public-health data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ProviderResult:
    """Standard provider output."""

    data: pd.DataFrame
    source_name: str
    metadata: dict[str, Any]


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""

    source_name: str = "base"

    @abstractmethod
    def fetch(self, **kwargs) -> ProviderResult:
        """Fetch data and return a standardized ProviderResult."""

    def validate_standard_columns(self, df: pd.DataFrame, date_col: str, value_col: str) -> None:
        """Validate that a provider result can be passed into the pipeline."""
        missing = [col for col in [date_col, value_col] if col not in df.columns]
        if missing:
            raise ValueError(f"Provider output missing required columns: {missing}")
