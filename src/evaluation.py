"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        raise ValueError("No finite paired values available for evaluation.")
    return true[mask], pred[mask]


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    true, pred = _clean_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    true, pred = _clean_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, excluding zero true values."""
    true, pred = _clean_arrays(y_true, y_pred)
    nonzero = true != 0
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error."""
    true, pred = _clean_arrays(y_true, y_pred)
    denominator = np.abs(true) + np.abs(pred)
    mask = denominator != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2 * np.abs(pred[mask] - true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, train_series: pd.Series, seasonal_period: int = 1) -> float:
    """Mean absolute scaled error versus naive or seasonal naive benchmark."""
    true, pred = _clean_arrays(y_true, y_pred)
    train = np.asarray(train_series, dtype=float)
    if len(train) <= seasonal_period:
        denom = np.mean(np.abs(np.diff(train)))
    else:
        denom = np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period]))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / denom)


def evaluate_forecast(y_true, y_pred, train_series: pd.Series | None = None, seasonal_period: int = 1) -> dict[str, float]:
    """Return common forecast metrics as a dictionary."""
    metrics = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }
    if train_series is not None:
        metrics["MASE"] = mase(y_true, y_pred, train_series, seasonal_period)
    return metrics
