from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import PipelineConfig, run_pipeline
from src.preprocessing import TimeSeriesConfig, prepare_time_series


def make_sample(path: Path) -> Path:
    dates = pd.date_range("2021-01-03", periods=140, freq="W-SUN")
    seasonal = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 52)
    trend = np.arange(len(dates)) * 0.03
    cases = np.maximum(0, seasonal + trend).round().astype(int)
    df = pd.DataFrame({"date": dates, "cases": cases, "disease": "Example flu", "county": "Example"})
    df.to_csv(path, index=False)
    return path


def test_prepare_time_series_weekly(tmp_path):
    input_path = make_sample(tmp_path / "sample.csv")
    df = pd.read_csv(input_path)
    series = prepare_time_series(df, TimeSeriesConfig(date_col="date", value_col="cases", frequency="weekly"))
    assert len(series) == 140
    assert series.index.is_monotonic_increasing
    assert series.min() >= 0


def test_pipeline_runs_without_sarima_for_fast_ci(tmp_path):
    input_path = make_sample(tmp_path / "sample.csv")
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        PipelineConfig(
            input_path=str(input_path),
            date_col="date",
            value_col="cases",
            horizon=4,
            seasonal_period=52,
            output_dir=str(output_dir),
            run_sarima=False,
            run_fourier=True,
            run_mlp=True,
        )
    )
    assert result["csv_path"].exists()
    assert result["xlsx_path"].exists()
    assert result["png_path"].exists()
    assert len(result["ensemble"]) == 4
