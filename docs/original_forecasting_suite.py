# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:36:27 2025

@author: KHOURYK
"""


import warnings
from typing import Optional, Tuple, Dict, Any, Union, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.neural_network import MLPRegressor
import matplotlib.dates as mdates

import seaborn as sns
import statsmodels.api as sm
from statsmodels.tools import add_constant
from scipy.stats import linregress, t
from datetime import datetime
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
sns.set(rc={'figure.figsize':(11.7,8.27)})
import os
import pypyodbc as odbc
from datetime import datetime



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _aicc(aic: float, n: int, k: int) -> float:
    # AICc = AIC + (2k(k+1)) / (n - k - 1)
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mase(y, yhat, m: int):
    """
    Mean Absolute Scaled Error vs seasonal naive with period m.
    If y length too short for seasonal naive, falls back to naive-1.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if len(y) <= m:
        denom = np.mean(np.abs(np.diff(y)))
    else:
        denom = np.mean(np.abs(y[m:] - y[:-m]))
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.mean(np.abs(y - yhat)) / denom)


def sarima_forecast(
    df: pd.DataFrame,
    date_col: str = "DTEVENT",
    value_col: str = "weekly_case_count",
    seasonal_period: int = 52,
    forecast_periods: int = 12,
    use_log1p: bool = True,
    # You can still force a specific model; otherwise we’ll select via CV.
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    # Validation settings
    holdout_weeks: Optional[int] = None,   # None -> 20% of data, at least 10, at most 52
    parsimony_lambda: float = 1.0,         # additional penalty (lambda * num_params) added to AICc
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    aggregation: str = "sum",
) -> Dict[str, Any]:
    """
    Fit a SARIMA model to weekly flu cases with out-of-sample validation to avoid overfitting.

    Model selection:
      1) Small, parsimonious grid of candidates
      2) Fit on train, forecast holdout, score by validation RMSE
      3) Tie-break with AICc + parsimony penalty
      4) Refit the chosen model on full history and produce forecast

    Returns
    -------
    dict with keys:
        'model' : fitted SARIMAXResults (refit on full data)
        'order' : (p,d,q)
        'seasonal_order' : (P,D,Q,m)
        'history' : pd.Series (weekly time series used to fit; transformed back if log used)
        'forecast' : pd.DataFrame with ['yhat','lower','upper'] (original scale)
        'val_rmse' : float (validation RMSE for selected model)
        'val_mase' : float (MASE on validation window vs seasonal naive)
        'aicc' : float (AICc of the refit model)
        'used_log1p' : bool
    """

    # --- 1) Prepare weekly time series (same interface you had) ---
    ts = (
        df[[date_col, value_col]]
        .dropna(subset=[date_col, value_col])
        .assign(**{date_col: pd.to_datetime(df[date_col])})
        .sort_values(date_col)
    )

    agg_func = np.sum if aggregation == "sum" else np.mean
    ts = (
        ts.groupby(date_col, as_index=False)[value_col].apply(agg_func)
        .set_index(date_col)
        .asfreq("D")
    )
    ts[value_col] = ts[value_col].fillna(0)

    weekly = ts[value_col].resample("W-SUN").apply(agg_func).fillna(0)

    if len(weekly) < (seasonal_period + 16):
        raise ValueError(
            f"Time series is too short: need at least ~{seasonal_period+16} weekly points; got {len(weekly)}."
        )

    # variance stabilization (helps reduce overfit on spikes)
    if use_log1p:
        y = np.log1p(weekly.clip(lower=0))
        inv = np.expm1
    else:
        y = weekly.astype(float)
        inv = lambda x: x

    n = len(y)

    # --- 2) Build parsimonious candidate grid (tiny & sane) ---
    # Keep it small: p,q in {0,1}, d in {0,1}; seasonal P,Q in {0,1}, D in {0,1}
    # Also include a couple of "workhorse" defaults.
    candidate_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 0)]
    candidate_seasonals = [(0, 1, 1, seasonal_period), (1, 1, 0, seasonal_period), (0, 1, 0, seasonal_period)]

    # If user passed explicit orders, skip selection and use them
    explicit = (order is not None) and (seasonal_order is not None)
    if explicit:
        candidate_pairs = [(order, seasonal_order)]
    else:
        candidate_pairs = [(o, so) for o in candidate_orders for so in candidate_seasonals]

    # --- 3) Create a holdout window for validation ---
    if holdout_weeks is None:
        holdout_weeks = int(max(10, min(52, np.floor(0.2 * n))))
    holdout_weeks = int(min(max(8, holdout_weeks), n - 8))  # keep some train history

    y_train = y.iloc[:-holdout_weeks]
    y_val = y.iloc[-holdout_weeks:]

    freq = weekly.index.freqstr or "W-SUN"

    def try_fit(endog, o, so):
        # parameter count k ~ #statespace params; statsmodels doesn't expose directly pre-fit.
        # We'll approximate by counting AR + MA + seasonal + variance.
        approx_k = sum(o) + sum(so[:3]) + 1
        model = SARIMAX(
            endog,
            order=o,
            seasonal_order=so,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            trend="n",
            freq=freq,
        ).fit(disp=False)
        # Use AICc + parsimony bias for tie-breaks / sanity
        aicc = _aicc(model.aic, n=len(endog), k=approx_k)
        aicc_pars = aicc + parsimony_lambda * approx_k
        return model, aicc_pars, aicc, approx_k

    best = {
        "pair": None,
        "val_rmse": np.inf,
        "val_mase": np.inf,
        "aicc_pars": np.inf,
        "aicc": np.inf,
        "k": None,
    }

    # --- 4) Score candidates on validation RMSE ---
    for o, so in candidate_pairs:
        try:
            mdl, aicc_pars_train, aicc_train, k = try_fit(y_train, o, so)
            # Forecast into validation window
            steps = len(y_val)
            pred = mdl.get_forecast(steps=steps)
            yhat_val = pred.predicted_mean
            rmse = _rmse(y_val.values, yhat_val.values)
            mase = _mase(y_val.values, yhat_val.values, m=seasonal_period)

            # Choose by RMSE first; tie-breaker is AICc + parsimony
            is_better = (rmse < best["val_rmse"]) or (
                np.isclose(rmse, best["val_rmse"]) and (aicc_pars_train < best["aicc_pars"])
            )
            if is_better:
                best.update(
                    {
                        "pair": (o, so),
                        "val_rmse": rmse,
                        "val_mase": mase,
                        "aicc_pars": aicc_pars_train,
                        "aicc": aicc_train,
                        "k": k,
                    }
                )
        except Exception:
            continue

    if best["pair"] is None:
        raise RuntimeError("No SARIMA candidate converged. Try relaxing constraints or disabling log1p.")

    # --- 5) Refit best candidate on FULL history and forecast ---
    o, so = best["pair"]
    final_model = SARIMAX(
        y,
        order=o,
        seasonal_order=so,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend="n",
        freq=freq,
    ).fit(disp=False)

    # Forecast forward
    fcst = final_model.get_forecast(steps=forecast_periods)
    mean_fcst = inv(fcst.predicted_mean)
    ci = fcst.conf_int(alpha=0.05)
    # conf_int is on transformed scale if use_log1p=True; invert bounds carefully
    if use_log1p:
        lower = inv(ci.iloc[:, 0].values)
        upper = inv(ci.iloc[:, 1].values)
    else:
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values

    forecast_df = pd.DataFrame(
        {"yhat": mean_fcst.values, "lower": lower, "upper": upper},
        index=mean_fcst.index,
    )
    forecast_df.index.name = "week_end"

    # Prepare history back on original scale if transformed
    history_out = inv(y)
    history_out.index = weekly.index  # keep weekly index

    # Compute AICc for the full fit (reporting only)
    approx_k_full = best["k"]
    aicc_full = _aicc(final_model.aic, n=len(y), k=approx_k_full)

    return {
        "model": final_model,
        "order": o,
        "seasonal_order": so,
        "history": history_out,
        "forecast": forecast_df,
        "val_rmse": best["val_rmse"],
        "val_mase": best["val_mase"],
        "aic": final_model.aic,
        "aicc": aicc_full,
        "used_log1p": bool(use_log1p),
    }




# def weekly_counts(df, id_col="MEDSISID", date_col="DTEVENT", start_date=None):
#     # Ensure date column is datetime
#     df[date_col] = pd.to_datetime(df[date_col])

#     # Filter for events on or after start_date
#     df = df[df[date_col] >= pd.to_datetime(start_date)]

#     # Drop duplicates to avoid double-counting same case/date
#     df = df[[id_col, date_col]].drop_duplicates()

#     # Set datetime index
#     df = df.set_index(date_col)

#     # Resample weekly (weeks ending Monday here) and count unique MEDSISIDs
#     weekly_df = (
#         df.resample("W-MON")[id_col]
#         .nunique()  # unique cases per week
#         .reset_index()
#         .rename(columns={id_col: "weekly_case_count"})
#     )

#     return weekly_df


def weekly_counts(df, id_col="MEDSISID", date_col="DTEVENT", start_date=None, end_date=None):
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])  # Remove rows with invalid dates

    # Apply date filtering if start_date or end_date are provided
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # Drop duplicates to avoid double-counting the same case/date
    df = df[[id_col, date_col]].drop_duplicates()

    # Set datetime index
    df = df.set_index(date_col)

    # Resample weekly (weeks ending Monday) and count unique IDs
    weekly_df = (
        df.resample("W-MON")[id_col]
        .nunique()
        .reset_index()
        .rename(columns={id_col: "weekly_case_count"})
    )

    return weekly_df




def plot_sarima_forecast(
    results: dict,
    title: str = "SARIMA Forecast",
    figsize=(10, 5),
    include_in_sample: bool = True,
    history_label: str = "Observed",
    fitted_label: str = "Fitted (in-sample)",
    forecast_label: str = "Forecast",
    ci_alpha: float = 0.2,
    grid: bool = True,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot observed weekly series, optional in-sample fitted values,
    and forecast with 95% confidence intervals (both shaded + dashed lines).

    Parameters
    ----------
    results : dict
        Output dict from sarima_forecast() with keys 'history', 'forecast', 'model'.
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    include_in_sample : bool
        If True, plot one-step-ahead in-sample predictions.
    history_label, fitted_label, forecast_label : str
        Legend labels.
    ci_alpha : float
        Transparency for the confidence interval shading.
    grid : bool
        Toggle background grid.
    save_path : str or None
        If provided, saves the figure to this path.
    show : bool
        If True, displays the plot with plt.show().
    """
    history = results["history"]           # pd.Series
    forecast_df = results["forecast"]      # DataFrame with ['yhat','lower','upper']
    model = results["model"]

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Observed history
    ax.plot(history.index, history.values, label=history_label, linewidth=1.8, color="tab:blue")

    # 2) In-sample fitted values (optional)
    if include_in_sample:
        try:
            fitted = model.get_prediction(start=history.index[0], end=history.index[-1]).predicted_mean
        except Exception:
            fitted = model.fittedvalues
        ax.plot(fitted.index, fitted.values, linestyle="--", linewidth=1.2,
                label=fitted_label, color="tab:green")

    # 3) Forecast mean + CI (shaded + dashed)
    ax.plot(forecast_df.index, forecast_df["yhat"], color="tab:orange", linewidth=2.0,
            label=forecast_label)
    ax.fill_between(
        forecast_df.index,
        forecast_df["lower"],
        forecast_df["upper"],
        color="tab:orange",
        alpha=ci_alpha,
        label="95% CI",
    )
    ax.plot(forecast_df.index, forecast_df["lower"], color="tab:orange",
            linestyle="--", linewidth=1.0)
    ax.plot(forecast_df.index, forecast_df["upper"], color="tab:orange",
            linestyle="--", linewidth=1.0)

    # Formatting
    ax.set_title(title, pad=12)
    ax.set_xlabel("Week end date")
    ax.set_ylabel("Weekly cases")
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    # Nice weekly date formatting
    try:
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
    except Exception:
        pass

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)



# --- Fourier Series model class ---
class FourierModel:
    def __init__(self, period, coeffs, n_harmonics):
        self.period = period
        self.coeffs = coeffs
        self.n_harmonics = n_harmonics

    def evaluate(self, t):
        P = self.period
        a0 = self.coeffs[0]
        y = a0 * np.ones_like(t, dtype=float)
        for k in range(1, self.n_harmonics + 1):
            ak = self.coeffs[2*k-1]
            bk = self.coeffs[2*k]
            w = 2*np.pi*k/P
            y += ak * np.cos(w*t) + bk * np.sin(w*t)
        return y

    def d1(self, t):
        P = self.period
        dydt = np.zeros_like(t, dtype=float)
        for k in range(1, self.n_harmonics + 1):
            ak = self.coeffs[2*k-1]
            bk = self.coeffs[2*k]
            w = 2*np.pi*k/P
            dydt += -ak*w*np.sin(w*t) + bk*w*np.cos(w*t)
        return dydt

    def d2(self, t):
        P = self.period
        d2ydt2 = np.zeros_like(t, dtype=float)
        for k in range(1, self.n_harmonics + 1):
            ak = self.coeffs[2*k-1]
            bk = self.coeffs[2*k]
            w = 2*np.pi*k/P
            d2ydt2 += -(ak*(w**2))*np.cos(w*t) - (bk*(w**2))*np.sin(w*t)
        return d2ydt2


# --- Helpers ---
def _to_numeric_time(series):
    ts = pd.to_datetime(series)
    t0 = ts.min()
    secs = (ts - t0).dt.total_seconds().to_numpy()
    return secs, ts

def _design_matrix(t, period, n_harmonics):
    cols = [np.ones_like(t, dtype=float)]
    for k in range(1, n_harmonics + 1):
        w = 2*np.pi*k/period
        cols.append(np.cos(w*t))
        cols.append(np.sin(w*t))
    return np.column_stack(cols)

def _fit_fourier(t, y, period, n_harmonics):
    X = _design_matrix(t, period, n_harmonics)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return FourierModel(period=period, coeffs=coeffs, n_harmonics=n_harmonics)


# --- Inflection detection ---
def _find_inflections(tg, d2):
    # zero-crossings of d2
    idx = np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0]
    return tg[idx]


# --- Main analysis function ---
def fourier_inflection_analysis(data, time_col, value_col, period=None, n_harmonics=5):
    df = data[[time_col, value_col]].dropna().copy()
    t_secs, ts_parsed = _to_numeric_time(df[time_col])
    y = df[value_col].to_numpy(dtype=float)
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # if no period given, use total span / 3 as fallback
    if period is None:
        span = (ts_parsed.max() - ts_parsed.min()).total_seconds()
        period = max(span/3.0, 1.0)

    model = _fit_fourier(t_secs, y_centered, period, n_harmonics)

    tg = np.linspace(t_secs.min(), t_secs.max(), 2000)
    yhat = model.evaluate(tg) + y_mean
    d1 = model.d1(tg)
    d2 = model.d2(tg)

    # Inflections
    xs = _find_inflections(tg, d2)
    ts_inflect = pd.to_datetime(xs, unit="s", origin=ts_parsed.min())
    yhat_inf = np.interp(xs, tg, yhat)

    # Plot 1: data + Fourier fit
    plt.figure()
    plt.plot(ts_parsed, y, 'o', label='Observed')
    plt.plot(pd.to_datetime(tg, unit="s", origin=ts_parsed.min()), yhat, label='Fourier fit')
    plt.scatter(ts_inflect, yhat_inf, marker='x', s=60, label='Inflections')
    plt.legend(); plt.title("Fourier Fit with Inflection Points")
    plt.show()

    # Plot 2: second derivative
    plt.figure()
    plt.plot(pd.to_datetime(tg, unit="s", origin=ts_parsed.min()), d2, label='Second derivative')
    plt.axhline(0, color='k', linestyle='--', lw=1)
    plt.scatter(ts_inflect, np.interp(xs, tg, d2), marker='x', s=60, label='Inflections')
    plt.legend(); plt.title("Second Derivative with Inflection Points")
    plt.show()

    # Plot 3: raw data only
    plt.figure()
    plt.plot(ts_parsed, y, 'o-', color='gray')
    plt.title("Raw Data")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.show()

    return pd.DataFrame({"time": ts_inflect, "yhat": yhat_inf})



def _make_weekly_series(df, time_col, value_col, aggregation="sum", week_ending="SUN"):
    """Convert irregular dates to weekly series with no gaps."""
    df = df[[time_col, value_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    agg = np.sum if aggregation == "sum" else np.mean
    daily = (
        df.groupby(time_col, as_index=False)[value_col].apply(agg)
          .set_index(time_col)
          .asfreq("D")
    )
    daily[value_col] = daily[value_col].fillna(0)
    weekly = daily[value_col].resample(f"W-{week_ending}").apply(agg).fillna(0)
    return weekly


def _to_numeric_time_weeks(index_like):
    """Return numeric time in weeks since first timestamp, plus the origin."""
    ts = pd.to_datetime(index_like)
    t0 = ts.min()
    weeks = (ts - t0) / np.timedelta64(1, "W")
    return weeks.to_numpy(dtype=float), t0

def fourier_fit_and_forecast(
    data,
    time_col: str = "DTEVENT",
    value_col: str = "weekly_case_count",
    period_weeks: float = 52.0,
    n_harmonics: int = 5,
    forecast_weeks: int = 12,
    train_last_weeks: int | None = None,
    aggregation: str = "sum",
    week_ending: str = "SUN",
    plot_title: str = "Fourier Forecast",
    ci_alpha: float = 0.2,
    grid: bool = True,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Fit a Fourier series to weekly data, forecast a specified number of future weeks,
    and plot observed, in-sample fit, and forecast with ±1.96σ confidence bands.

    Returns
    -------
    dict with keys:
      'model'          : FourierModel
      'history'        : pd.Series (weekly)
      'fitted'         : pd.Series (in-sample fitted on history)
      'forecast'       : pd.DataFrame with ['yhat','lower','upper'] indexed by week end
      'resid_std'      : float (in-sample residual std)
      'period_weeks'   : float
      'n_harmonics'    : int
    """
    # --- Step 1: Prepare weekly data ---
    weekly = _make_weekly_series(data, time_col, value_col, aggregation=aggregation, week_ending=week_ending)

    # Optional rolling window for recent data
    if train_last_weeks is not None and train_last_weeks > 0 and train_last_weeks < len(weekly):
        cutoff = weekly.index.max() - pd.Timedelta(weeks=train_last_weeks - 1)
        weekly_train = weekly.loc[weekly.index >= cutoff].copy()
    else:
        weekly_train = weekly.copy()

    # --- Step 2: Numeric time (weeks) ---
    t_hist_weeks, t0 = _to_numeric_time_weeks(weekly_train.index)
    y = weekly_train.to_numpy(dtype=float)
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # --- Step 3: Fit Fourier model ---
    model = _fit_fourier(t_hist_weeks * 7 * 24 * 3600, y_centered,
                         period_weeks * 7 * 24 * 3600, n_harmonics)

    # --- Step 4: Dense evaluation for plotting ---
    last_hist_date = weekly.index.max()
    future_end = last_hist_date + pd.Timedelta(weeks=forecast_weeks)
    ts_dense = pd.date_range(t0, future_end, freq="D")
    t_dense_secs = (ts_dense - t0).total_seconds().to_numpy(dtype=float)
    yhat_dense = model.evaluate(t_dense_secs) + y_mean

    # --- Step 5: In-sample fitted values (for entire history) ---
    ts_hist_all = weekly.index
    t_hist_all_secs = (ts_hist_all - t0).total_seconds().to_numpy(dtype=float)
    yhat_hist_all = model.evaluate(t_hist_all_secs) + y_mean
    fitted_series = pd.Series(yhat_hist_all, index=ts_hist_all, name="fitted")

    # --- Step 6: Residual std (for CI) ---
    fitted_train = pd.Series(model.evaluate(t_hist_weeks * 7 * 24 * 3600) + y_mean,
                             index=weekly_train.index)
    resid = weekly_train - fitted_train
    resid_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else float(np.nanstd(resid))

    # --- Step 7: Build forecast table ---
    ts_future_weeks = pd.date_range(last_hist_date + pd.Timedelta(days=1),
                                    future_end, freq=f"W-{week_ending}")
    if len(ts_future_weeks) == 0:
        ts_future_weeks = pd.DatetimeIndex([last_hist_date + pd.Timedelta(weeks=1)])

    t_future_secs = (ts_future_weeks - t0).total_seconds().to_numpy(dtype=float)
    yhat_future = model.evaluate(t_future_secs) + y_mean
    lower = yhat_future - 1.96 * resid_std
    upper = yhat_future + 1.96 * resid_std
    forecast_df = pd.DataFrame({"yhat": yhat_future, "lower": lower, "upper": upper},
                               index=ts_future_weeks)
    forecast_df.index.name = "week_end"

    # --- Step 8: Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekly.index, weekly.values, label="Observed", linewidth=1.8)
    ax.plot(fitted_series.index, fitted_series.values, linestyle="--", linewidth=1.2, label="Fourier fit")
    ax.plot(forecast_df.index, forecast_df["yhat"], linewidth=2.0, label="Forecast")
    ax.fill_between(forecast_df.index, forecast_df["lower"], forecast_df["upper"],
                    alpha=ci_alpha, label="95% CI", linewidth=0)
    ax.axvline(last_hist_date, linestyle=":", linewidth=1.0)
    ax.set_title(plot_title)
    ax.set_xlabel("Week end date")
    ax.set_ylabel(value_col)
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Step 9: Print and return results ---
    print("\n📊 Fourier Forecast Summary")
    print(f"Training points: {len(weekly_train)}  |  Total history: {len(weekly)} weeks")
    print(f"Period (weeks): {period_weeks:.2f}  |  Harmonics: {n_harmonics}")
    print(f"Residual Std (σ): {resid_std:.3f}")
    print(f"Forecast weeks: {forecast_weeks}\n")
    print("🔹 Forecast preview:")
    print(forecast_df.head())

    return {
        "model": model,
        "history": weekly,
        "fitted": fitted_series,
        "forecast": forecast_df,
        "resid_std": resid_std,
        "period_weeks": period_weeks,
        "n_harmonics": n_harmonics,
    }


def neural_forecast_mlp(df, time_col="DTEVENT", value_col="weekly_case_count",
                        lookback=12, forecast_weeks=12):
    # 1. Prepare weekly numeric array
    df = df[[time_col, value_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    y = df[value_col].to_numpy(dtype=float)
    X, Y = [], []
    for i in range(lookback, len(y)):
        X.append(y[i-lookback:i])
        Y.append(y[i])
    X, Y = np.array(X), np.array(Y)

    # 2. Fit MLP
    model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation='relu', solver='adam',
                         learning_rate_init=0.001,
                         max_iter=500, random_state=42)
    model.fit(X, Y)

    # 3. In-sample fitted
    fitted_vals = np.concatenate([np.full(lookback, np.nan),
                                  model.predict(X)])
    fitted_series = pd.Series(fitted_vals, index=df[time_col], name="fitted")

    # 4. Recursive forecast
    preds = []
    last_win = y[-lookback:].copy()
    for _ in range(forecast_weeks):
        p = model.predict(last_win.reshape(1, -1))[0]
        preds.append(p)
        last_win = np.concatenate([last_win[1:], [p]])
    resid_std = float(np.nanstd(Y - model.predict(X)))
    lower = np.array(preds) - 1.96 * resid_std
    upper = np.array(preds) + 1.96 * resid_std

    # 5. Build forecast DataFrame
    last_date = df[time_col].max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=7),
                               periods=forecast_weeks, freq="W-SUN")
    forecast_df = pd.DataFrame({"yhat": preds, "lower": lower, "upper": upper},
                               index=future_idx)
    forecast_df.index.name = "week_end"

    # 6. Plot
    plt.figure(figsize=(10,5))
    plt.plot(df[time_col], y, label="Observed")
    plt.plot(fitted_series.index, fitted_series.values, "--", label="Fitted")
    plt.plot(forecast_df.index, forecast_df["yhat"], label="Forecast")
    plt.fill_between(forecast_df.index, forecast_df["lower"], forecast_df["upper"],
                     alpha=0.2, label="95% CI")
    plt.legend(); plt.title("MLP Neural Net Forecast"); plt.show()

    print("🔹 Forecast preview:")
    print(forecast_df.head())

    return {
        "model": model,
        "history": pd.Series(y, index=df[time_col]),
        "fitted": fitted_series,
        "forecast": forecast_df,
        "resid_std": resid_std,
    }


def plot_ensemble_from_forecasts(
    df: pd.DataFrame,
    time_col: str = "DTEVENT",
    value_col: str = "weekly_case_count",
    forecasts: Union[Dict[str, pd.DataFrame], Iterable[pd.DataFrame]] = None,
    labels: Optional[Iterable[str]] = None,
    ensemble_method: str = "median",      # "median" or "mean"
    ensemble_ci: Tuple[float, float] = (0.10, 0.90),  # quantiles across model yhat
    show_individual_cis: bool = True,
    ci_alpha: float = 0.12,
    title: str = "Ensemble Forecasts",
    figsize: Tuple[int, int] = (11, 6),
    grid: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot original series + multiple forecast DataFrames (yhat/lower/upper) on one chart
    and add an ensemble line computed across model point-forecasts.

    Parameters
    ----------
    df : DataFrame with [time_col, value_col]
    forecasts : dict(name -> forecast_df) OR iterable of forecast_dfs
        Each forecast_df must have index as datetime-like and columns ['yhat','lower','upper'].
    labels : optional list of names (only used if 'forecasts' is an iterable, not a dict)
    ensemble_method : 'median' or 'mean'
    ensemble_ci : (low_q, high_q) quantiles across model yhat to draw ensemble band
    """

    if forecasts is None:
        raise ValueError("Provide one or more forecast DataFrames via 'forecasts'.")

    # Normalize to dict[name -> df]
    if isinstance(forecasts, dict):
        fc_dict = forecasts
    else:
        if labels is None:
            labels = [f"Model{i+1}" for i, _ in enumerate(forecasts)]
        fc_dict = {name: f for name, f in zip(labels, forecasts)}

    # Prepare observed series
    df = df[[time_col, value_col]].dropna().copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    y_obs = pd.Series(df[value_col].to_numpy(dtype=float), index=df[time_col], name="Observed")

    # Union of all forecast horizons
    all_idx = pd.Index([])
    for name, f in fc_dict.items():
        if not {"yhat","lower","upper"}.issubset(f.columns):
            raise ValueError(f"Forecast '{name}' must have columns: yhat, lower, upper")
        all_idx = all_idx.union(pd.to_datetime(f.index))
    all_idx = all_idx.sort_values()

    # Stack yhat (and optionally CIs)
    yhat_wide = pd.DataFrame(index=all_idx)
    for name, f in fc_dict.items():
        tmp = f.copy()
        tmp.index = pd.to_datetime(tmp.index)
        yhat_wide[name] = tmp.reindex(all_idx)["yhat"]

    # Ensemble point forecasts
    if ensemble_method.lower() == "mean":
        ens_yhat = yhat_wide.mean(axis=1, skipna=True)
    elif ensemble_method.lower() == "median":
        ens_yhat = yhat_wide.median(axis=1, skipna=True)
    else:
        raise ValueError("ensemble_method must be 'mean' or 'median'.")

    # Ensemble CI from model spread (quantiles across model yhat)
    qlo, qhi = ensemble_ci
    ens_lower = yhat_wide.quantile(qlo, axis=1, interpolation="linear")
    ens_upper = yhat_wide.quantile(qhi, axis=1, interpolation="linear")
    ensemble_df = pd.DataFrame({"yhat": ens_yhat, "lower": ens_lower, "upper": ens_upper}, index=all_idx)
    ensemble_df.index.name = "week_end"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_obs.index, y_obs.values, linewidth=2.0, label="Observed")

    for name, f in fc_dict.items():
        f2 = f.copy()
        f2.index = pd.to_datetime(f2.index)
        ax.plot(f2.index, f2["yhat"].values, linewidth=1.8, label=f"{name}")
        if show_individual_cis:
            ax.fill_between(f2.index, f2["lower"].values, f2["upper"].values,
                            alpha=ci_alpha, linewidth=0)

    # Ensemble overlay (thicker line + band)
    ax.plot(ensemble_df.index, ensemble_df["yhat"].values, linewidth=2.6,
            label=f"Ensemble ({ensemble_method})")
    ax.fill_between(ensemble_df.index, ensemble_df["lower"].values, ensemble_df["upper"].values,
                    alpha=max(ci_alpha, 0.18), linewidth=0,
                    label=f"Ensemble CI {int((ensemble_ci[1]-ensemble_ci[0])*100)}%")

    # Vertical line at last observed date (handy visual)
    ax.axvline(y_obs.index.max(), linestyle=":", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Week end date")
    ax.set_ylabel(value_col)
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return ensemble_df




#------------------------------------Data Upload and Prep---------------------------------------------------------------------#

medsis = pd.read_csv('C:/Users/khouryk/Downloads/MEDSIS Extract 20251021.txt',sep='\t', low_memory=False)  

fluA = medsis.loc[medsis['MORBSPC']== 'A']
fluB = medsis.loc[medsis['MORBSPC']== 'B']
fluA_H1 = medsis.loc[medsis['MORBSPC']== 'A/H1']
fluA_H3 = medsis.loc[medsis['MORBSPC']== 'A/H3']



new_fluA_full_hist_12wk = weekly_counts(fluA, start_date="2020-10-01", end_date = None)

new_fluA_season22_12wk = weekly_counts(fluA, start_date="2020-10-01", end_date = "2022-10-01" )



new_fluA_season24_12wk = weekly_counts(fluA, start_date="2022-10-01", end_date = "2024-10-01" )


new_fluA_season25_12wk = weekly_counts(fluA, start_date="2023-10-01", end_date = None )





new_fluB = weekly_counts(fluB, start_date="2023-10-01")


new_fluB_full_hist_12wk = weekly_counts(fluB, start_date="2020-10-01", end_date = None)






new_fluA_H1 = weekly_counts(fluA_H1)
new_fluA_H3 = weekly_counts(fluA_H3)

medsis = pd.read_csv('C:/Users/khouryk/Downloads/MEDSIS Extract 20251022.txt',sep='\t', low_memory=False)  
cocci = weekly_counts(medsis, start_date="2018-01-01")


cocci2 = weekly_counts(medsis, start_date="2021-01-01", end_date = "2024-03-01")



#------------------------------------SARIMA Forecasting---------------------------------------------------------------------#
results_full_hist_12wk = sarima_forecast(
    new_fluA_full_hist_12wk,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results_full_hist_12wk, title="FluA Cases: SARIMA 12-week Forecast", include_in_sample=True)


results_season22_12wk = sarima_forecast(
    new_fluA_season22_12wk,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results_season22_12wk, title="FluA Cases: SARIMA 12-week Forecast", include_in_sample=True)


results_season24_12wk = sarima_forecast(
    new_fluA_season24_12wk,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results_season24_12wk, title="FluA Cases: SARIMA 12-week Forecast", include_in_sample=True)



results_season25_12wk = sarima_forecast(
    new_fluA_season25_12wk,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results_season25_12wk, title="FluA Cases: SARIMA 12-week Forecast", include_in_sample=True)











results2_1_1 = sarima_forecast(
    new_fluA,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results2_1_1, title="FluA Cases: SARIMA 12-week Forecast", include_in_sample=True)



results3 = sarima_forecast(
    cocci,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=3     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results3, title="Coccidioidomycosis Cases: SARIMA 12-week Forecast", include_in_sample=True)




results3_2 = sarima_forecast(
    cocci2,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    seasonal_period=52,
    forecast_periods=12,
    use_log1p=False,          # turn off if your series is already variance-stable
    parsimony_lambda=1     # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results3_2, title="Coccidioidomycosis Cases: SARIMA 12-week Forecast", include_in_sample=True)












results3 = sarima_forecast(
    new_fluB_full_hist_12wk,
    date_col="DTEVENT",
    value_col="weekly_case_count",
    order=(1, 1, 1),              # your chosen non-seasonal ARIMA order
    seasonal_order=(1, 1, 1, 52), # <<–– your custom seasonal order
    seasonal_period=52,           # period (monthly seasonality)
    forecast_periods=12,          # number of periods to forecast
    use_log1p=False,               # optional: stabilizes variance
    parsimony_lambda=3.0,      # bump to 2–3 if you still see overfitting
)
#print("Chosen:", results["order"], results["seasonal_order"], "val_RMSE:", results["val_rmse"], "MASE:", results["val_mase"])
plot_sarima_forecast(results3, title="Flu Cases: SARIMA 12-week Forecast", include_in_sample=True)



#------------------------------------Fourier Series Forecasting---------------------------------------------------------------------#
inflections_full_hist_12wk = fourier_fit_and_forecast(
    new_fluA_full_hist_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Flu B: Fourier 12-week Forecast"
)


inflections_season22_12wk = fourier_fit_and_forecast(
    new_fluA_season22_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Flu B: Fourier 12-week Forecast"
)



inflections_season24_12wk = fourier_fit_and_forecast(
    new_fluA_season24_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Flu B: Fourier 12-week Forecast"
)


inflections_season25_12wk = fourier_fit_and_forecast(
    new_fluA_season25_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Flu B: Fourier 12-week Forecast"
)






inflections = fourier_fit_and_forecast(
    new_fluB_full_hist_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Flu B: Fourier 12-week Forecast"
)


inflections_cocci = fourier_fit_and_forecast(
    cocci,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Coccidioidomycosis: Fourier 12-week Forecast"
)


inflections_cocci = fourier_fit_and_forecast(
    cocci2,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    period_weeks=52,
    n_harmonics=5,
    forecast_weeks=12,
    train_last_weeks=156,
    plot_title="Coccidioidomycosis: Fourier 12-week Forecast"
)





inflections = fourier_inflection_analysis(new_fluA, "DTEVENT", "weekly_case_count")


#------------------------------------Nueral Network Forecasting---------------------------------------------------------------------#


# Example:
out_full_hist_12wk = neural_forecast_mlp(new_fluA_full_hist_12wk, "DTEVENT", "weekly_case_count")


out_full_season22_12wk = neural_forecast_mlp(new_fluA_season22_12wk, "DTEVENT", "weekly_case_count")


out_full_season24_12wk = neural_forecast_mlp(new_fluA_season24_12wk, "DTEVENT", "weekly_case_count")

out_full_season25_12wk = neural_forecast_mlp(new_fluA_season25_12wk, "DTEVENT", "weekly_case_count")


out_cocci = neural_forecast_mlp(cocci2, "DTEVENT", "weekly_case_count")



#------------------------------------Ensemble Forecasting---------------------------------------------------------------------#


ens_hist_12wk = plot_ensemble_from_forecasts(
    df=new_fluA_full_hist_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    forecasts={
        "SARIMA": results_full_hist_12wk["forecast"],
        "Fourier": inflections_full_hist_12wk["forecast"],
        "LSTM": out_full_hist_12wk["forecast"],   # or TF LSTM
    },
    ensemble_method="median",
    ensemble_ci=(0.10, 0.90),
    title="Flu A H1 Cases — Ensemble of Models"
)

ens_season22_12wk = plot_ensemble_from_forecasts(
    df=new_fluA_season22_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    forecasts={
        "SARIMA": results_season22_12wk["forecast"],
        "Fourier": inflections_season22_12wk["forecast"],
        "LSTM": out_full_season22_12wk["forecast"],   # or TF LSTM
    },
    ensemble_method="median",
    ensemble_ci=(0.10, 0.90),
    title="Flu Cases — Ensemble of Models"
)



ens_season24_12wk = plot_ensemble_from_forecasts(
    df=new_fluA_season24_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    forecasts={
        "SARIMA": results_season24_12wk["forecast"],
        "Fourier": inflections_season24_12wk["forecast"],
        "LSTM": out_full_season24_12wk["forecast"],   # or TF LSTM
    },
    ensemble_method="median",
    ensemble_ci=(0.10, 0.90),
    title="Flu Cases — Ensemble of Models"
)


ens_season25_12wk = plot_ensemble_from_forecasts(
    df=new_fluA_season25_12wk,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    forecasts={
        "SARIMA": results_season25_12wk["forecast"],
        "Fourier": inflections_season25_12wk["forecast"],
        "LSTM": out_full_season25_12wk["forecast"],   # or TF LSTM
    },
    ensemble_method="median",
    ensemble_ci=(0.10, 0.90),
    title="Flu Cases — Ensemble of Models"
)





ens_cocci = plot_ensemble_from_forecasts(
    df=cocci2,
    time_col="DTEVENT",
    value_col="weekly_case_count",
    forecasts={
        "SARIMA": results3_2["forecast"],
        "Fourier": inflections_cocci["forecast"],
        "LSTM": out_cocci["forecast"],   # or TF LSTM
    },
    ensemble_method="median",
    ensemble_ci=(0.10, 0.90),
    title="Coccidioidomycosis — Ensemble of Models"
)


#------------------------------------Forecasting Output---------------------------------------------------------------------#


ens.to_csv('C:/Users/khouryk/Desktop/flua_cases_forecast_22OCT2025_12wk.csv', sep=',', index=True)

ens_cocci.to_csv('C:/Users/khouryk/Desktop/Coccidioidomycosis_cases_forecast_12wk.csv', sep=',', index=True)


