"""Forecast model implementations: SARIMA, Fourier regression, and MLP neural net."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .features import fourier_design_matrix, future_index, make_lagged_matrix, numeric_time_index
from .utils import get_frequency_alias


@dataclass
class ForecastResult:
    """Standard output for all forecasting models."""

    model_name: str
    history: pd.Series
    forecast: pd.DataFrame
    fitted: pd.Series | None = None
    model: object | None = None
    metadata: dict = field(default_factory=dict)


def _aicc(aic: float, n: int, k: int) -> float:
    """Corrected Akaike Information Criterion."""
    if n - k - 1 <= 0:
        return float("inf")
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


def _inverse_log1p(values) -> np.ndarray:
    """Invert log1p with clipping to avoid numeric overflow in unstable intervals."""
    return np.expm1(np.clip(np.asarray(values, dtype=float), -50, 20))


def _clip_forecast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cases should not be negative; clip model outputs to zero."""
    out = frame.copy()
    for col in ["yhat", "lower", "upper"]:
        if col in out:
            out[col] = out[col].astype(float).clip(lower=0)
    return out


def sarima_forecast(
    series: pd.Series,
    horizon: int = 12,
    seasonal_period: int = 52,
    use_log1p: bool = True,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    holdout_periods: int | None = None,
    parsimony_lambda: float = 1.0,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> ForecastResult:
    """Fit a parsimonious SARIMA model and forecast future case counts."""
    y_original = series.astype(float).asfreq(series.index.freq or pd.infer_freq(series.index))
    y_original = y_original.fillna(0).clip(lower=0)
    if len(y_original) < max(12, seasonal_period // 2 + 8):
        raise ValueError(
            f"SARIMA needs a longer time series. Got {len(y_original)} observations; "
            f"try reducing seasonal_period or using Fourier/MLP only."
        )

    y = np.log1p(y_original) if use_log1p else y_original.copy()
    frequency = y_original.index.freqstr or pd.infer_freq(y_original.index) or "W-SUN"

    candidate_orders = [order] if order else [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 0)]
    candidate_seasonals = (
        [seasonal_order]
        if seasonal_order
        else [
            (0, 1, 1, seasonal_period),
            (1, 1, 0, seasonal_period),
            (0, 1, 0, seasonal_period),
            (0, 0, 1, seasonal_period),
        ]
    )

    if holdout_periods is None:
        holdout_periods = int(max(4, min(26, round(0.2 * len(y)))))
    holdout_periods = int(min(max(3, holdout_periods), len(y) - 4))
    train = y.iloc[:-holdout_periods]
    valid = y.iloc[-holdout_periods:]

    best: dict[str, object] = {"score": float("inf")}
    for sarima_order in candidate_orders:
        for sarima_seasonal in candidate_seasonals:
            try:
                approx_k = sum(sarima_order) + sum(sarima_seasonal[:3]) + 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    fitted = SARIMAX(
                        train,
                        order=sarima_order,
                        seasonal_order=sarima_seasonal,
                        trend="n",
                        freq=frequency,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                        simple_differencing=True,
                    ).fit(disp=False, maxiter=75)
                pred = fitted.get_forecast(steps=len(valid)).predicted_mean
                rmse = float(np.sqrt(np.mean((valid.to_numpy() - pred.to_numpy()) ** 2)))
                aicc_penalized = _aicc(fitted.aic, len(train), approx_k) + parsimony_lambda * approx_k
                score = rmse + 0.0001 * aicc_penalized
                if score < best["score"]:
                    best = {
                        "score": score,
                        "order": sarima_order,
                        "seasonal_order": sarima_seasonal,
                        "validation_rmse_transformed": rmse,
                        "aicc_train": aicc_penalized,
                    }
            except Exception:
                continue

    if "order" not in best:
        raise RuntimeError("No SARIMA model converged. Try a longer series or simpler seasonal period.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        final_model = SARIMAX(
            y,
            order=best["order"],
            seasonal_order=best["seasonal_order"],
            trend="n",
            freq=frequency,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            simple_differencing=True,
        ).fit(disp=False, maxiter=75)

    forecast_obj = final_model.get_forecast(steps=horizon)
    conf_int = forecast_obj.conf_int(alpha=0.05)
    if use_log1p:
        forecast_mean = pd.Series(_inverse_log1p(forecast_obj.predicted_mean), index=forecast_obj.predicted_mean.index)
        lower = _inverse_log1p(conf_int.iloc[:, 0])
        upper = _inverse_log1p(conf_int.iloc[:, 1])
    else:
        forecast_mean = forecast_obj.predicted_mean
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
    forecast = pd.DataFrame({"yhat": forecast_mean, "lower": lower, "upper": upper})
    forecast.index.name = "date"

    raw_fitted = final_model.fittedvalues
    fitted_arr = _inverse_log1p(raw_fitted) if use_log1p else np.asarray(raw_fitted, dtype=float)
    fitted_index = y_original.index[-len(fitted_arr) :]
    fitted_values = pd.Series(fitted_arr, index=fitted_index, name="fitted").reindex(y_original.index)

    metadata = {
        "order": best["order"],
        "seasonal_order": best["seasonal_order"],
        "aic": float(final_model.aic),
        "aicc": _aicc(final_model.aic, len(y), sum(best["order"]) + sum(best["seasonal_order"][:3]) + 1),
        "used_log1p": bool(use_log1p),
        "validation_rmse_transformed": best["validation_rmse_transformed"],
    }

    return ForecastResult(
        model_name="SARIMA",
        history=y_original,
        fitted=fitted_values.clip(lower=0),
        forecast=_clip_forecast_frame(forecast),
        model=final_model,
        metadata=metadata,
    )


def fourier_forecast(
    series: pd.Series,
    horizon: int = 12,
    seasonal_period: int = 52,
    n_harmonics: int = 5,
) -> ForecastResult:
    """Fit a Fourier regression model for seasonal case-count forecasting."""
    y = series.astype(float).fillna(0).clip(lower=0)
    if len(y) < 2 * n_harmonics + 3:
        raise ValueError("Fourier model needs more observations or fewer harmonics.")

    t = numeric_time_index(y.index)
    centered = y.to_numpy() - float(np.mean(y))
    x = fourier_design_matrix(t, seasonal_period, n_harmonics)
    coeffs, *_ = np.linalg.lstsq(x, centered, rcond=None)
    fitted_values = x @ coeffs + float(np.mean(y))
    residuals = y.to_numpy() - fitted_values
    resid_std = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 2 else float(np.nanstd(residuals))

    freq = y.index.freqstr or pd.infer_freq(y.index) or "W-SUN"
    idx_future = future_index(y.index[-1], freq, horizon)
    t_future = np.arange(len(y), len(y) + horizon, dtype=float)
    future_design = fourier_design_matrix(t_future, seasonal_period, n_harmonics)
    yhat = future_design @ coeffs + float(np.mean(y))
    forecast = pd.DataFrame(
        {
            "yhat": yhat,
            "lower": yhat - 1.96 * resid_std,
            "upper": yhat + 1.96 * resid_std,
        },
        index=idx_future,
    )
    forecast.index.name = "date"

    return ForecastResult(
        model_name="Fourier",
        history=y,
        fitted=pd.Series(fitted_values, index=y.index, name="fitted").clip(lower=0),
        forecast=_clip_forecast_frame(forecast),
        model={"coefficients": coeffs},
        metadata={"seasonal_period": seasonal_period, "n_harmonics": n_harmonics, "resid_std": resid_std},
    )


def mlp_forecast(
    series: pd.Series,
    horizon: int = 12,
    lookback: int = 12,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    random_state: int = 42,
) -> ForecastResult:
    """Fit a lightweight scikit-learn MLP and recursively forecast ahead."""
    y = series.astype(float).fillna(0).clip(lower=0)
    lookback = min(lookback, max(2, len(y) // 3))
    x, target = make_lagged_matrix(y, lookback)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(target.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=25,
    )
    model.fit(x_scaled, y_scaled)

    fitted_scaled = model.predict(x_scaled)
    fitted_tail = y_scaler.inverse_transform(fitted_scaled.reshape(-1, 1)).ravel()
    fitted_values = np.concatenate([np.full(lookback, np.nan), fitted_tail])
    fitted_series = pd.Series(fitted_values, index=y.index, name="fitted")

    predictions = []
    window = y.to_numpy(dtype=float)[-lookback:].copy()
    for _ in range(horizon):
        window_scaled = x_scaler.transform(window.reshape(1, -1))
        pred_scaled = model.predict(window_scaled)
        pred = float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0])
        pred = max(pred, 0.0)
        predictions.append(pred)
        window = np.concatenate([window[1:], [pred]])

    residuals = target - fitted_tail
    resid_std = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 2 else float(np.nanstd(residuals))
    freq = y.index.freqstr or pd.infer_freq(y.index) or "W-SUN"
    idx_future = future_index(y.index[-1], freq, horizon)
    pred_arr = np.asarray(predictions)
    forecast = pd.DataFrame(
        {
            "yhat": pred_arr,
            "lower": pred_arr - 1.96 * resid_std,
            "upper": pred_arr + 1.96 * resid_std,
        },
        index=idx_future,
    )
    forecast.index.name = "date"

    return ForecastResult(
        model_name="MLP",
        history=y,
        fitted=fitted_series.clip(lower=0),
        forecast=_clip_forecast_frame(forecast),
        model=model,
        metadata={"lookback": lookback, "resid_std": resid_std},
    )


def run_selected_models(
    series: pd.Series,
    horizon: int,
    seasonal_period: int,
    model_names: Iterable[str] = ("sarima", "fourier", "mlp"),
    n_harmonics: int = 5,
    lookback: int = 12,
    use_log1p: bool = True,
) -> dict[str, ForecastResult]:
    """Run selected model names and return successful results."""
    results: dict[str, ForecastResult] = {}
    errors: dict[str, str] = {}
    requested = {name.lower().strip() for name in model_names}

    if "sarima" in requested:
        try:
            results["SARIMA"] = sarima_forecast(series, horizon, seasonal_period, use_log1p=use_log1p)
        except Exception as exc:
            errors["SARIMA"] = str(exc)

    if "fourier" in requested:
        try:
            results["Fourier"] = fourier_forecast(series, horizon, seasonal_period, n_harmonics=n_harmonics)
        except Exception as exc:
            errors["Fourier"] = str(exc)

    if "mlp" in requested:
        try:
            results["MLP"] = mlp_forecast(series, horizon, lookback=lookback)
        except Exception as exc:
            errors["MLP"] = str(exc)

    if not results:
        raise RuntimeError(f"No models completed successfully. Errors: {errors}")

    for result in results.values():
        result.metadata.setdefault("model_errors", errors)
    return results
