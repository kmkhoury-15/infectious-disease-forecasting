"""Feature engineering helpers for Fourier and neural network models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_design_matrix(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Create a Fourier design matrix with intercept, cosine, and sine terms."""
    if period <= 0:
        raise ValueError("period must be positive.")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1.")

    cols = [np.ones_like(t, dtype=float)]
    for k in range(1, n_harmonics + 1):
        omega = 2 * np.pi * k / period
        cols.append(np.cos(omega * t))
        cols.append(np.sin(omega * t))
    return np.column_stack(cols)


def numeric_time_index(index: pd.Index) -> np.ndarray:
    """Convert a DatetimeIndex to elapsed time steps from the first observation."""
    idx = pd.to_datetime(index)
    return np.arange(len(idx), dtype=float)


def make_lagged_matrix(series: pd.Series, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Create X/y arrays for one-step-ahead supervised learning."""
    values = series.to_numpy(dtype=float)
    if lookback < 2:
        raise ValueError("lookback must be >= 2.")
    if len(values) <= lookback:
        raise ValueError(f"Need more than lookback={lookback} observations; got {len(values)}.")

    x_rows, y_rows = [], []
    for idx in range(lookback, len(values)):
        x_rows.append(values[idx - lookback : idx])
        y_rows.append(values[idx])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def future_index(last_date: pd.Timestamp, frequency_alias: str, horizon: int) -> pd.DatetimeIndex:
    """Build a future DatetimeIndex after the final observed date."""
    return pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=frequency_alias)[1:]
