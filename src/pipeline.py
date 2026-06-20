"""End-to-end forecasting pipeline and command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import infer_schema, load_case_data, normalize_column_names
from .ensemble import ensemble_forecasts
from .evaluation import evaluate_forecast
from .models import run_selected_models
from .preprocessing import TimeSeriesConfig, prepare_time_series, train_test_split_series
from .utils import ensure_directory, safe_filename, validate_positive_int
from .visualization import plot_forecast


@dataclass
class PipelineConfig:
    """User-facing forecasting pipeline settings."""

    input_path: str
    date_col: str | None = None
    value_col: str | None = None
    disease_col: str | None = None
    geography_col: str | None = None
    disease_value: str | None = None
    geography_value: str | None = None
    frequency: str = "weekly"
    horizon: int = 12
    seasonal_period: int = 52
    train_test_split: float = 0.2
    aggregation: str = "sum"
    fill_method: str = "zero"
    smooth_outliers: bool = False
    ensemble_method: str = "weighted_mean"
    sarima_weight: float = 1.0
    fourier_weight: float = 1.0
    mlp_weight: float = 1.0
    n_harmonics: int = 5
    lookback: int = 12
    output_dir: str = "outputs"
    run_sarima: bool = True
    run_fourier: bool = True
    run_mlp: bool = True
    use_log1p: bool = True


def _resolve_columns(df: pd.DataFrame, config: PipelineConfig) -> PipelineConfig:
    inferred = infer_schema(df)
    config.date_col = config.date_col or inferred["date_col"]
    config.value_col = config.value_col or inferred["value_col"]
    config.disease_col = config.disease_col or inferred["disease_col"]
    config.geography_col = config.geography_col or inferred["geography_col"]
    if not config.date_col or not config.value_col:
        raise ValueError(
            "Could not infer required date/value columns. Provide --date-col and --value-col. "
            f"Available columns: {list(df.columns)}"
        )
    return config


def _selected_model_names(config: PipelineConfig) -> list[str]:
    models: list[str] = []
    if config.run_sarima:
        models.append("sarima")
    if config.run_fourier:
        models.append("fourier")
    if config.run_mlp:
        models.append("mlp")
    return models or ["fourier", "mlp"]


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """Run the full pipeline from input file through saved outputs."""
    config.horizon = validate_positive_int(config.horizon, "horizon")
    config.seasonal_period = validate_positive_int(config.seasonal_period, "seasonal_period")

    raw = normalize_column_names(load_case_data(config.input_path))
    config = _resolve_columns(raw, config)

    ts_config = TimeSeriesConfig(
        date_col=config.date_col,
        value_col=config.value_col,
        frequency=config.frequency,
        aggregation=config.aggregation,
        disease_col=config.disease_col,
        geography_col=config.geography_col,
        disease_value=config.disease_value,
        geography_value=config.geography_value,
        fill_method=config.fill_method,  # type: ignore[arg-type]
        smooth_outliers=config.smooth_outliers,
    )
    series = prepare_time_series(raw, ts_config)

    try:
        train, test = train_test_split_series(series, config.train_test_split)
        evaluation_results = {}
        eval_models = run_selected_models(
            train,
            horizon=len(test),
            seasonal_period=min(config.seasonal_period, max(2, len(train) // 2)),
            model_names=_selected_model_names(config),
            n_harmonics=config.n_harmonics,
            lookback=config.lookback,
            use_log1p=config.use_log1p,
        )
        eval_ensemble = ensemble_forecasts(
            eval_models,
            method=config.ensemble_method,
            weights={"SARIMA": config.sarima_weight, "Fourier": config.fourier_weight, "MLP": config.mlp_weight},
        )
        aligned = eval_ensemble.reindex(test.index)
        evaluation_results["Ensemble"] = evaluate_forecast(test, aligned["yhat"], train, config.seasonal_period)
    except Exception as exc:
        evaluation_results = {"warning": f"Evaluation skipped: {exc}"}

    model_results = run_selected_models(
        series,
        horizon=config.horizon,
        seasonal_period=config.seasonal_period,
        model_names=_selected_model_names(config),
        n_harmonics=config.n_harmonics,
        lookback=config.lookback,
        use_log1p=config.use_log1p,
    )
    ensemble = ensemble_forecasts(
        model_results,
        method=config.ensemble_method,
        weights={"SARIMA": config.sarima_weight, "Fourier": config.fourier_weight, "MLP": config.mlp_weight},
    )

    output_root = ensure_directory(config.output_dir)
    forecasts_dir = ensure_directory(output_root / "forecasts")
    plots_dir = ensure_directory(output_root / "plots")
    stem = safe_filename(Path(config.input_path).stem)
    csv_path = forecasts_dir / f"{stem}_forecast.csv"
    xlsx_path = forecasts_dir / f"{stem}_forecast.xlsx"
    png_path = plots_dir / f"{stem}_forecast.png"

    export = ensemble.reset_index().rename(columns={"date": "forecast_date"})
    export.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        export.to_excel(writer, sheet_name="ensemble_forecast", index=False)
        pd.DataFrame(series).reset_index().to_excel(writer, sheet_name="history", index=False)
        pd.DataFrame(evaluation_results).to_excel(writer, sheet_name="evaluation")

    fig = plot_forecast(
        history=series,
        model_results=model_results,
        ensemble=ensemble,
        title="Infectious Disease Case Forecast",
        save_path=png_path,
    )

    return {
        "config": asdict(config),
        "history": series,
        "model_results": model_results,
        "ensemble": ensemble,
        "evaluation": evaluation_results,
        "csv_path": csv_path,
        "xlsx_path": xlsx_path,
        "png_path": png_path,
        "figure": fig,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run infectious disease case forecasting.")
    parser.add_argument("--input", required=True, dest="input_path", help="Path to CSV/XLSX/XLS file.")
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--value-col", default=None)
    parser.add_argument("--disease-col", default=None)
    parser.add_argument("--geography-col", default=None)
    parser.add_argument("--disease-value", default=None)
    parser.add_argument("--geography-value", default=None)
    parser.add_argument("--frequency", default="weekly", choices=["daily", "weekly", "quarterly", "yearly"])
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seasonal-period", type=int, default=52)
    parser.add_argument("--train-test-split", type=float, default=0.2)
    parser.add_argument("--aggregation", default="sum", choices=["sum", "mean"])
    parser.add_argument("--fill-method", default="zero", choices=["zero", "ffill", "interpolate"])
    parser.add_argument("--smooth-outliers", action="store_true")
    parser.add_argument("--ensemble-method", default="weighted_mean", choices=["weighted_mean", "mean", "median"])
    parser.add_argument("--sarima-weight", type=float, default=1.0)
    parser.add_argument("--fourier-weight", type=float, default=1.0)
    parser.add_argument("--mlp-weight", type=float, default=1.0)
    parser.add_argument("--n-harmonics", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-sarima", action="store_false", dest="run_sarima")
    parser.add_argument("--no-fourier", action="store_false", dest="run_fourier")
    parser.add_argument("--no-mlp", action="store_false", dest="run_mlp")
    parser.add_argument("--no-log1p", action="store_false", dest="use_log1p")
    return parser.parse_args(args=args)


def main(args: Iterable[str] | None = None) -> None:
    namespace = parse_args(args)
    result = run_pipeline(PipelineConfig(**vars(namespace)))
    print("Forecast completed.")
    print(f"CSV:  {result['csv_path']}")
    print(f"Excel: {result['xlsx_path']}")
    print(f"Plot: {result['png_path']}")
    print("Evaluation:", result["evaluation"])


if __name__ == "__main__":
    main()
