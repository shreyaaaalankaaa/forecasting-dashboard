"""Streamlit dashboard: upload a CSV (or use the demo data), pick a target, compare forecasters."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    MODELS,
    add_lag_features,
    add_time_features,
    chronological_split,
    make_demo_dataset,
    results_table,
    train_and_evaluate,
)

st.set_page_config(page_title="Forecasting Dashboard", layout="wide")
st.title("Forecasting Dashboard")
st.caption(
    "Upload a CSV, choose what to predict, and compare classical forecasting models "
    "against a naive baseline on a chronological (past → future) split."
)

# ---------------- Data ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Dataset", ["Demo: hourly electricity demand (synthetic)", "Upload CSV"])
    upload = st.file_uploader("CSV file", type=["csv"], disabled=source != "Upload CSV")

if source == "Upload CSV":
    if upload is None:
        st.info("Upload a CSV with at least two numeric columns to begin.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
else:
    raw = make_demo_dataset()

st.subheader("Data preview")
st.dataframe(raw.head(10), use_container_width=True)

# ---------------- Setup ----------------
with st.sidebar:
    st.header("Setup")
    columns = list(raw.columns)
    guess_time = next((c for c in columns if "time" in c.lower() or "date" in c.lower()), None)
    time_column = st.selectbox("Time column (optional)", ["None"] + columns,
                               index=(columns.index(guess_time) + 1) if guess_time else 0)
    time_column = None if time_column == "None" else time_column

data = raw
if time_column:
    try:
        data = add_time_features(raw, time_column)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
if len(numeric) < 2:
    st.error("The dataset needs at least two numeric columns (features and a target).")
    st.stop()

with st.sidebar:
    default_target = numeric.index("demand_mw") if "demand_mw" in numeric else len(numeric) - 1
    target = st.selectbox("Target to predict", numeric, index=default_target)
    lags = []
    if time_column:
        lag_text = st.text_input("Lag features (previous target values)", "1, 24",
                                 help="e.g. 1, 24 adds the value 1 step and 24 steps earlier as features")
        try:
            lags = sorted({int(x) for x in lag_text.replace(" ", "").split(",") if x})
        except ValueError:
            st.warning("Lags must be whole numbers separated by commas; ignoring them.")
            lags = []
    if lags:
        data = add_lag_features(data, target, lags)
    candidates = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c != target]
    features = st.multiselect("Feature columns", candidates, default=candidates)
    test_fraction = st.slider("Test set (latest rows)", 0.1, 0.4, 0.2, 0.05)
    chosen_models = st.multiselect("Models", list(MODELS), default=list(MODELS))

if not features:
    st.warning("Choose at least one feature column.")
    st.stop()

# ---------------- Train ----------------
try:
    results = train_and_evaluate(data, target, features, test_fraction, time_ordered=bool(time_column),
                                 models=chosen_models)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

_, test = chronological_split(data[features + [target]].apply(pd.to_numeric, errors="coerce").dropna(), test_fraction)
best = min(results[1:], key=lambda r: r.mae) if len(results) > 1 else results[0]

st.subheader("Model comparison")
st.markdown(f"Best model on the test period: **{best.name}**")
cols = st.columns(3)
cols[0].metric("MAE", f"{best.mae:,.2f}")
cols[1].metric("RMSE", f"{best.rmse:,.2f}")
cols[2].metric("R²", f"{best.r2:.3f}")
st.dataframe(results_table(results), use_container_width=True, hide_index=True)
st.caption("Error reduction is measured against the naive baseline's MAE. A model that cannot beat it is not useful.")

# ---------------- Charts ----------------
x_axis = data.loc[test.index, time_column] if time_column else test.index
show = st.selectbox("Show predictions for", [r.name for r in results], index=[r.name for r in results].index(best.name))
picked = next(r for r in results if r.name == show)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_axis, y=test[target], name="Actual", line={"color": "#1f2937", "width": 1.6}))
fig.add_trace(go.Scatter(x=x_axis, y=picked.predictions, name=f"Predicted ({picked.name})",
                         line={"color": "#2563eb", "width": 1.6}))
fig.update_layout(title=f"Actual vs predicted {target} on the test period", height=420,
                  legend={"orientation": "h", "y": 1.1}, margin={"t": 70})
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
residuals = test[target].to_numpy() - picked.predictions
with left:
    hist = px.histogram(x=residuals, nbins=40, title="Prediction errors (actual − predicted)",
                        labels={"x": f"Error in {target}"})
    hist.add_vline(x=0, line_dash="dash")
    st.plotly_chart(hist, use_container_width=True)
with right:
    with_importance = [r for r in results if r.importances is not None]
    importance_source = picked if picked.importances is not None else (with_importance[0] if with_importance else None)
    if importance_source is not None:
        top = importance_source.importances.head(10)[::-1]
        bars = px.bar(x=top.values, y=top.index, orientation="h",
                      title=f"Most important features ({importance_source.name})",
                      labels={"x": "Importance", "y": ""})
        st.plotly_chart(bars, use_container_width=True)
    else:
        st.info("Feature importance is available for the tree models.")

download = pd.DataFrame({"actual": test[target].to_numpy(), "predicted": picked.predictions,
                         "error": np.round(residuals, 4)})
st.download_button("Download test-period predictions (CSV)", download.to_csv(index=False), "predictions.csv", "text/csv")
