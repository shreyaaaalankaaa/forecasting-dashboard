"""Forecasting logic used by the dashboard, kept separate from the UI so it can be tested.

Everything here is classical machine learning (scikit-learn). Each model is compared
with a naive baseline, so a reported error always means something: a model is only
useful if it beats "predict the previous value" (or "predict the average").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "Linear regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Random forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient boosting": lambda: GradientBoostingRegressor(random_state=42),
}


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    mape: float | None
    predictions: np.ndarray
    importances: pd.Series | None = None


def make_demo_dataset(hours: int = 24 * 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic hourly electricity demand with daily and weekly cycles plus weather.

    Clearly synthetic: it exists so the dashboard runs without a download.
    """
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2026-01-01", periods=hours, freq="h")
    hour = timestamp.hour.to_numpy()
    weekday = timestamp.dayofweek.to_numpy()
    day = np.arange(hours) / 24
    temperature = 24 + 6 * np.sin(2 * np.pi * (hour - 15) / 24) + 3 * np.sin(2 * np.pi * day / 60) + rng.normal(0, 1.2, hours)
    humidity = np.clip(65 - 1.5 * (temperature - 24) + rng.normal(0, 5, hours), 20, 100)
    daily = 120 * np.exp(-((hour - 19) ** 2) / 8) + 60 * np.exp(-((hour - 9) ** 2) / 6)
    weekend = np.where(weekday >= 5, -70, 0)
    cooling = 9 * np.clip(temperature - 26, 0, None)
    trend = 0.15 * day
    demand = 420 + daily + weekend + cooling + trend + rng.normal(0, 14, hours)
    return pd.DataFrame({
        "timestamp": timestamp,
        "temperature_c": temperature.round(2),
        "humidity_pct": humidity.round(1),
        "is_weekend": (weekday >= 5).astype(int),
        "demand_mw": demand.round(1),
    })


def add_time_features(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Turn a timestamp into hour, weekday and month columns a model can use."""
    out = df.copy()
    stamps = pd.to_datetime(out[time_column], errors="coerce")
    if stamps.isna().all():
        raise ValueError(f"Column '{time_column}' could not be read as dates.")
    out = out.loc[stamps.notna()].copy()
    stamps = stamps.loc[out.index]
    out["hour"] = stamps.dt.hour
    out["weekday"] = stamps.dt.dayofweek
    out["month"] = stamps.dt.month
    return out.sort_values(time_column).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    """Previous values of the target as features. Rows without full history are dropped."""
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out.dropna().reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the past, test on the future. Shuffling time series would leak the future."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 5% and 50%.")
    cut = int(len(df) * (1 - test_fraction))
    if cut < 10 or len(df) - cut < 5:
        raise ValueError("Not enough rows for a train/test split. Upload more data.")
    return df.iloc[:cut], df.iloc[cut:]


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    nonzero = np.abs(actual) > 1e-9
    if nonzero.sum() < len(actual) * 0.9:
        return None  # undefined when many actual values are zero
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def score(name: str, actual: np.ndarray, predicted: np.ndarray, importances: pd.Series | None = None) -> ModelResult:
    return ModelResult(
        name=name,
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        r2=float(r2_score(actual, predicted)),
        mape=_mape(actual, predicted),
        predictions=np.asarray(predicted, dtype=float),
        importances=importances,
    )


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str, time_ordered: bool) -> ModelResult:
    """Previous value for time-ordered data (persistence), otherwise the training mean."""
    actual = test[target].to_numpy(dtype=float)
    if time_ordered:
        history = pd.concat([train[target].iloc[-1:], test[target]]).to_numpy(dtype=float)
        predicted = history[:-1]
        return score("Naive: previous value", actual, predicted)
    predicted = np.full(len(test), float(train[target].mean()))
    return score("Naive: training mean", actual, predicted)


def train_and_evaluate(df: pd.DataFrame, target: str, features: list[str], test_fraction: float = 0.2,
                       time_ordered: bool = True, models: list[str] | None = None) -> list[ModelResult]:
    """Fit each model on the earlier rows and score it on the later rows."""
    if target in features:
        raise ValueError("The target column cannot also be a feature.")
    if not features:
        raise ValueError("Choose at least one feature column.")
    data = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 30:
        raise ValueError("Need at least 30 complete numeric rows.")
    train, test = chronological_split(data, test_fraction)
    results = [naive_baseline(train, test, target, time_ordered)]
    for name in models or list(MODELS):
        model = MODELS[name]()
        model.fit(train[features], train[target])
        estimator = model[-1] if hasattr(model, "steps") else model
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = pd.Series(estimator.feature_importances_, index=features).sort_values(ascending=False)
        results.append(score(name, test[target].to_numpy(dtype=float), model.predict(test[features]), importances))
    return results


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    baseline_mae = results[0].mae
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "MAE": round(r.mae, 3),
            "RMSE": round(r.rmse, 3),
            "R²": round(r.r2, 3),
            "MAPE %": None if r.mape is None else round(r.mape, 2),
            "Error reduction vs naive": "baseline" if r is results[0] else f"{(1 - r.mae / baseline_mae) * 100:+.1f}%",
        })
    return pd.DataFrame(rows)
