import numpy as np
import pandas as pd
import pytest

from forecasting import (
    add_lag_features,
    add_time_features,
    chronological_split,
    make_demo_dataset,
    results_table,
    train_and_evaluate,
)


def demo_frame():
    data = add_time_features(make_demo_dataset(hours=24 * 30), "timestamp")
    return add_lag_features(data, "demand_mw", [1, 24])


def test_split_keeps_time_order():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    train, test = chronological_split(df, 0.2)
    assert train["x"].max() < test["x"].min()
    assert len(test) == 20


def test_models_beat_naive_baseline_on_demo_data():
    df = demo_frame()
    features = [c for c in df.columns if c not in ("timestamp", "demand_mw")]
    results = train_and_evaluate(df, "demand_mw", features, 0.2, time_ordered=True)
    baseline = results[0]
    assert baseline.name.startswith("Naive")
    best = min(results[1:], key=lambda r: r.mae)
    assert best.mae < baseline.mae
    assert len(best.predictions) == int(len(df) - int(len(df) * 0.8))


def test_lag_features_drop_rows_without_history():
    df = pd.DataFrame({"y": np.arange(10.0)})
    lagged = add_lag_features(df, "y", [3])
    assert len(lagged) == 7
    assert (lagged["y"] - lagged["y_lag_3"]).eq(3).all()


def test_target_cannot_be_a_feature():
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    with pytest.raises(ValueError):
        train_and_evaluate(df, "b", ["a", "b"])


def test_results_table_compares_with_baseline():
    df = demo_frame()
    features = ["temperature_c", "hour", "weekday", "demand_mw_lag_1"]
    table = results_table(train_and_evaluate(df, "demand_mw", features, models=["Linear regression"]))
    assert list(table["Model"]) == ["Naive: previous value", "Linear regression"]
    assert table.iloc[0]["Error reduction vs naive"] == "baseline"
