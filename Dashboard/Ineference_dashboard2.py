import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Engine Health Inference & Forecast Dashboard",
    layout="wide"
)

st.title("🚀 Engine Health Inference & Forecast Dashboard")
st.write(
    "Run inference using CNN / LLM / Hybrid models, "
    "understand sensor contribution, and forecast engine health for next 10–15 days."
)

# ================= FEATURE CONFIG ================= #
selected_sensors = [2, 3, 4, 7, 11, 12, 15, 20, 21]
expected_feature_cols = (
    [f'OpSet{i}' for i in range(1, 4)] +
    [f'Sensor{i}' for i in selected_sensors]
)

# ================= UTILITY ================= #
def get_existing_features(data):
    return [c for c in expected_feature_cols if c in data.columns]

# ================= MODEL PLACEHOLDERS ================= #
def base_predict(data):
    features = get_existing_features(data)
    data_pred = data[features].copy()
    data_pred["RUL_label"] = np.random.choice([0, 1], len(data_pred))
    sensor_contrib = {f"Sensor{i}": np.random.rand() for i in selected_sensors}
    return data_pred, sensor_contrib

def cnn_predict(data): return base_predict(data)
def llm_predict(data): return base_predict(data)
def hybrid_predict(data): return base_predict(data)

# ================= FORECAST WITH MAINTENANCE ================= #
def generate_forecast(data, days, threshold, maintenance_done):
    seed = int(data.sum().sum() * 1000) % 2**32
    np.random.seed(seed)

    dates, scores, status = [], [], []
    failed = False

    for i in range(days):
        dates.append(datetime.today() + timedelta(days=i))

        if failed and not maintenance_done:
            score = np.random.uniform(0.2, threshold - 0.05)
            scores.append(score)
            status.append("Fail")
            continue

        if failed and maintenance_done:
            score = np.random.uniform(0.7, 0.9)
            scores.append(score)
            status.append("Healthy")
            failed = False
            continue

        score = np.random.uniform(0.3, 1.0)
        scores.append(score)

        if score < threshold:
            status.append("Fail")
            failed = True
        else:
            status.append("Healthy")

    return pd.DataFrame({
        "Date": dates,
        "Health Score": scores,
        "Status": status
    })

# ================= SIDEBAR ================= #
st.sidebar.header("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["CNN Only", "LLM Only", "Hybrid (CNN+LLM)"]
)

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 10, 15, 12)

maintenance_done = st.sidebar.checkbox(
    "Assume maintenance after failure?",
    value=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# ================= MODEL THRESHOLD ================= #
if model_option == "CNN Only":
    threshold = 0.55
elif model_option == "LLM Only":
    threshold = 0.60
else:
    threshold = 0.58

# ================= MAIN ================= #
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write(list(data.columns))
    st.dataframe(data.head())

    # -------- Summary -------- #
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Model", model_option)

    # -------- Prediction -------- #
    st.subheader("🔍 Inference Output")
    if st.button("Run Prediction"):
        if model_option == "CNN Only":
            prediction, sensor_contrib = cnn_predict(data)
        elif model_option == "LLM Only":
            prediction, sensor_contrib = llm_predict(data)
        else:
            prediction, sensor_contrib = hybrid_predict(data)

        display_cols = (
            [c for c in expected_feature_cols if c in prediction.columns]
            + ["RUL_label"]
        )

        st.dataframe(prediction[display_cols])

        st.subheader("📊 Sensor Contribution (Reason for Prediction)")
        sensor_df = pd.DataFrame(
            sensor_contrib.items(),
            columns=["Sensor", "Contribution"]
        ).sort_values("Contribution", ascending=False)

        st.bar_chart(sensor_df.set_index("Sensor"))

    # -------- Forecast -------- #
    st.subheader("📈 Engine Health Forecast (10–15 Days)")
    if st.button("Generate Forecast"):
        forecast = generate_forecast(
            data,
            forecast_days,
            threshold,
            maintenance_done
        )

        st.dataframe(forecast)

        fail_days = forecast[forecast["Status"] == "Fail"]
        fail_dates = fail_days["Date"].dt.strftime("%Y-%m-%d").tolist()

        c1, c2 = st.columns(2)
        c1.metric("Healthy Days", forecast["Status"].value_counts().get("Healthy", 0))
        c2.metric("Failure Days", len(fail_dates))

        # -------- Plot -------- #
        fig, ax = plt.subplots(figsize=(10, 4))

        # Health score line
        ax.plot(
            forecast["Date"],
            forecast["Health Score"],
            marker="o",
            linestyle="-",
            label="Health Score"
        )

        # Red dots for failures
        ax.scatter(
            fail_days["Date"],
            fail_days["Health Score"],
            s=100,
            c="red",
            label="Failure"
        )

        # Annotate failures
        for _, row in fail_days.iterrows():
            ax.text(
                row["Date"],
                row["Health Score"] + 0.03,
                "Fail",
                color="red",
                fontsize=9,
                ha="center"
            )

        ax.set_title("Engine Health Forecast (Red dots = Failure)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Health Score")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        # -------- Alerts -------- #
        if fail_dates:
            st.warning(f"⚠ Failure expected on: {', '.join(fail_dates)}")
        else:
            st.success("✅ No failures expected in forecast horizon")

        if maintenance_done:
            st.info(
                "ℹ Assumption: Maintenance is performed after failure, "
                "so engine health recovers on subsequent days."
            )
        else:
            st.info(
                "ℹ Assumption: No maintenance is performed; "
                "engine remains in failed condition."
            )

else:
    st.info("Upload a CSV file to begin inference.")
