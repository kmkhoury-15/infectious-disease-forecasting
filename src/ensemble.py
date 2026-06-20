"""Forecast ensembling utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .models import ForecastResult


def normalize_weights(model_names: list[str], weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Normalize provided model weights; default to equal weights."""
    if not weights:
        return {name: 1.0 / len(model_names) for name in model_names}
    values = {name: float(weights.get(name, 0.0)) for name in model_names}
    total = sum(max(v, 0.0) for v in values.values())
    if total <= 0:
        return {name: 1.0 / len(model_names) for name in model_names}
    return {name: max(v, 0.0) / total for name, v in values.items()}


def ensemble_forecasts(
    forecasts: Mapping[str, ForecastResult | pd.DataFrame],
    method: str = "weighted_mean",
    weights: Mapping[str, float] | None = None,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """Combine model forecasts into one ensemble forecast dataframe."""
    if not forecasts:
        raise ValueError("At least one forecast is required.")

    frames: dict[str, pd.DataFrame] = {}
    for name, result in forecasts.items():
        frame = result.forecast if isinstance(result, ForecastResult) else result
        required = {"yhat", "lower", "upper"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Forecast '{name}' must have columns: {sorted(required)}")
        copy = frame.copy()
        copy.index = pd.to_datetime(copy.index)
        frames[name] = copy

    all_index = pd.Index([])
    for frame in frames.values():
        all_index = all_index.union(frame.index)
    all_index = pd.DatetimeIndex(all_index).sort_values()

    yhat_wide = pd.DataFrame({name: frame.reindex(all_index)["yhat"] for name, frame in frames.items()})

    method = method.lower()
    if method == "median":
        yhat = yhat_wide.median(axis=1, skipna=True)
    elif method == "mean":
        yhat = yhat_wide.mean(axis=1, skipna=True)
    elif method == "weighted_mean":
        norm_weights = normalize_weights(list(yhat_wide.columns), weights)
        weight_vector = np.asarray([norm_weights[name] for name in yhat_wide.columns], dtype=float)
        yhat = yhat_wide.mul(weight_vector, axis=1).sum(axis=1, skipna=True)
    else:
        raise ValueError("method must be one of: weighted_mean, mean, median.")

    lower = yhat_wide.quantile(lower_quantile, axis=1, interpolation="linear")
    upper = yhat_wide.quantile(upper_quantile, axis=1, interpolation="linear")
    ensemble = pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper}, index=all_index)
    ensemble.index.name = "date"
    return ensemble.clip(lower=0)
