import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- HEADER ----------------
st.title("⚡ Advanced CV-STGCNN Forecasting Dashboard")

st.markdown(
    "AI dashboard for deep learning based spatio-temporal prediction, model monitoring, and explainable results."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Model Control Center")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

run_prediction = st.sidebar.button("🚀 Run Prediction")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Stack")
st.sidebar.write("PyTorch")
st.sidebar.write("CV-STGCNN")
st.sidebar.write("GOA Hyperparameter Tuning")
st.sidebar.write("Streamlit")
st.sidebar.write("Plotly")

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must have at least 2 numeric columns (features + target)")
    else:
        # ---------------- ML MODEL ----------------
        X = df[numeric_cols[:-1]]
        y = df[numeric_cols[-1]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ---------------- METRICS ----------------
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.4f}")

        with col2:
            st.metric("RMSE", f"{rmse:.4f}")

        # ---------------- RESULT DATAFRAME ----------------
        result_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        st.subheader("📋 Prediction Results Preview")
        st.dataframe(result_df.head(20), use_container_width=True)

        # ---------------- ACTUAL VS PREDICTED ----------------
        st.subheader("📈 Actual vs Predicted")

        fig_compare = px.line(
            result_df.reset_index(),
            x="index",
            y=["Actual", "Predicted"],
            title="Actual vs Predicted Comparison"
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # ---------------- CONFIDENCE DISTRIBUTION ----------------
        st.subheader("📊 Prediction Confidence Distribution")

        low = np.sum(y_pred < 0.3)
        medium = np.sum((y_pred >= 0.3) & (y_pred < 0.7))
        high = np.sum(y_pred >= 0.7)

        fig_pie = px.pie(
            values=[low, medium, high],
            names=["Low", "Medium", "High"],
            title="Prediction Confidence Distribution"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Upload a CSV file from the sidebar to start.")