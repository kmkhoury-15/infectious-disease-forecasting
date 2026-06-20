"""Time-series cleaning, validation, aggregation, and optional smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .utils import get_aggregation_function, get_frequency_alias

FillMethod = Literal["zero", "ffill", "interpolate"]


@dataclass(frozen=True)
class TimeSeriesConfig:
    """Configuration used to convert case data into a regular time series."""

    date_col: str
    value_col: str
    frequency: str = "weekly"
    aggregation: str = "sum"
    disease_col: str | None = None
    geography_col: str | None = None
    disease_value: str | None = None
    geography_value: str | None = None
    fill_method: FillMethod = "zero"
    start_date: str | None = None
    end_date: str | None = None
    smooth_outliers: bool = False


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a helpful error if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {list(df.columns)}")


def filter_optional_dimensions(df: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Filter to a single disease/geography if optional dimension values are provided."""
    out = df.copy()
    if config.disease_col and config.disease_value:
        validate_columns(out, [config.disease_col])
        out = out[out[config.disease_col].astype(str) == str(config.disease_value)]
    if config.geography_col and config.geography_value:
        validate_columns(out, [config.geography_col])
        out = out[out[config.geography_col].astype(str) == str(config.geography_value)]
    return out


def impute_series(series: pd.Series, method: FillMethod = "zero") -> pd.Series:
    """Impute missing values after resampling to a regular time grid."""
    if method == "zero":
        return series.fillna(0)
    if method == "ffill":
        return series.ffill().fillna(0)
    if method == "interpolate":
        return series.interpolate(limit_direction="both").fillna(0)
    raise ValueError("fill_method must be one of: zero, ffill, interpolate.")


def winsorize_outliers(series: pd.Series, z_threshold: float = 4.0) -> pd.Series:
    """Clip extreme values using a robust median/MAD z-score rule."""
    values = series.astype(float)
    median = values.median()
    mad = float(np.median(np.abs(values - median)))
    if mad == 0 or np.isnan(mad):
        return values
    robust_z = 0.6745 * (values - median) / mad
    out = values.copy()
    out[robust_z > z_threshold] = values[robust_z <= z_threshold].max()
    out[robust_z < -z_threshold] = values[robust_z >= -z_threshold].min()
    return out


def prepare_time_series(df: pd.DataFrame, config: TimeSeriesConfig) -> pd.Series:
    """Convert raw case data into a clean, regular pandas Series."""
    validate_columns(df, [config.date_col, config.value_col])
    out = filter_optional_dimensions(df, config)
    if out.empty:
        raise ValueError("No rows remain after optional disease/geography filtering.")

    out = out[[config.date_col, config.value_col]].copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="coerce")
    out[config.value_col] = pd.to_numeric(out[config.value_col], errors="coerce")
    out = out.dropna(subset=[config.date_col, config.value_col]).sort_values(config.date_col)

    if out.empty:
        raise ValueError("No valid rows remain after date/value parsing.")

    if config.start_date:
        out = out[out[config.date_col] >= pd.to_datetime(config.start_date)]
    if config.end_date:
        out = out[out[config.date_col] <= pd.to_datetime(config.end_date)]
    if out.empty:
        raise ValueError("No rows remain after start/end date filtering.")

    get_aggregation_function(config.aggregation)  # validates the user option
    agg_name = "sum" if config.aggregation == "sum" else "mean"
    frequency_alias = get_frequency_alias(config.frequency)

    out = out.groupby(config.date_col, as_index=True)[config.value_col].agg(agg_name).sort_index()
    full_index = pd.date_range(out.index.min(), out.index.max(), freq=frequency_alias)
    series = out.resample(frequency_alias).agg(agg_name).reindex(full_index)
    series = impute_series(series, config.fill_method)
    if config.smooth_outliers:
        series = winsorize_outliers(series)

    series.name = "cases"
    series.index.name = "date"
    return series.astype(float).clip(lower=0)


def train_test_split_series(series: pd.Series, test_size: float | int = 0.2) -> tuple[pd.Series, pd.Series]:
    """Split a time series into train and test windows without shuffling."""
    if len(series) < 12:
        raise ValueError("At least 12 observations are recommended before model evaluation.")

    if isinstance(test_size, float):
        n_test = max(1, int(round(len(series) * test_size)))
    else:
        n_test = int(test_size)
    n_test = min(max(1, n_test), len(series) - 2)
    return series.iloc[:-n_test], series.iloc[-n_test:]
