"""Forecast visualization functions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models import ForecastResult


def plot_forecast(
    history: pd.Series,
    model_results: Mapping[str, ForecastResult] | None,
    ensemble: pd.DataFrame | None,
    title: str = "Infectious Disease Forecast",
    save_path: str | Path | None = None,
    show_individual_models: bool = True,
):
    """Plot observed data, optional individual models, and ensemble forecast."""
    fig, ax = plt.subplots(figsize=(12, 6))
    history = history.astype(float)
    ax.plot(history.index, history.values, linewidth=2.0, label="Observed")

    if model_results and show_individual_models:
        for name, result in model_results.items():
            fc = result.forecast
            ax.plot(fc.index, fc["yhat"], linewidth=1.2, linestyle="--", label=name)

    if ensemble is not None:
        ax.plot(ensemble.index, ensemble["yhat"], linewidth=2.6, label="Ensemble forecast")
        ax.fill_between(
            ensemble.index,
            ensemble["lower"].to_numpy(dtype=float),
            ensemble["upper"].to_numpy(dtype=float),
            alpha=0.2,
            linewidth=0,
            label="Ensemble interval",
        )

    ax.axvline(history.index.max(), linestyle=":", linewidth=1.0, label="Forecast start")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best")
    fig.autofmt_xdate()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
