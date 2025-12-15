import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------- Expected Features ---------------- #
# Selected sensors indices (9 sensors)
selected_sensors = [2, 3, 4, 7, 11, 12, 15, 20, 21]

# Generate expected feature column names
expected_feature_cols = [f'op_setting_{i}' for i in range(1, 4)] + [
    f'sensor_measurement_{i}' for i in selected_sensors
]

# ---------------- Model Functions ---------------- #

def get_existing_features(data):
    """Return only columns that exist in uploaded CSV"""
    return [c for c in expected_feature_cols if c in data.columns]

# CNN
def cnn_predict(data):
    features = get_existing_features(data)
    data = data[features].copy()
    # Simulate prediction
    data['RUL_label'] = np.random.choice([0,1], len(data))
    # Simulate sensor importance contribution
    sensor_contrib = {f'sensor_measurement_{i}': np.random.rand() for i in selected_sensors}
    return data, sensor_contrib

def cnn_forecast(data, days):
    seed = int(data.sum().sum()*1000) % 2**32
    np.random.seed(seed)
    dates = [datetime.today() + timedelta(days=i) for i in range(days)]
    score = np.random.uniform(0.3,1,days)
    status = ["Fail" if s < 0.55 else "Healthy" for s in score]
    return pd.DataFrame({"Date": dates, "Health Score": score, "Status": status})

# LLM
def llm_predict(data):
    features = get_existing_features(data)
    data = data[features].copy()
    data['RUL_label'] = np.random.choice([0,1], len(data))
    sensor_contrib = {f'sensor_measurement_{i}': np.random.rand() for i in selected_sensors}
    return data, sensor_contrib

def llm_forecast(data, days):
    seed = int(data.sum().sum()*1000 + 1) % 2**32
    np.random.seed(seed)
    dates = [datetime.today() + timedelta(days=i) for i in range(days)]
    score = np.random.uniform(0.4,1,days)
    status = ["Fail" if s < 0.60 else "Healthy" for s in score]
    return pd.DataFrame({"Date": dates, "Health Score": score, "Status": status})

# Hybrid
def hybrid_predict(data):
    features = get_existing_features(data)
    data = data[features].copy()
    data['RUL_label'] = np.random.choice([0,1], len(data))
    sensor_contrib = {f'sensor_measurement_{i}': np.random.rand() for i in selected_sensors}
    return data, sensor_contrib

def hybrid_forecast(data, days):
    seed = int(data.sum().sum()*1000 + 2) % 2**32
    np.random.seed(seed)
    dates = [datetime.today() + timedelta(days=i) for i in range(days)]
    score = np.random.uniform(0.35,1,days)
    status = ["Fail" if s < 0.58 else "Healthy" for s in score]
    return pd.DataFrame({"Date": dates, "Health Score": score, "Status": status})

# ---------------- Streamlit Page Config ---------------- #
st.set_page_config(
    page_title="Engine Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Engine Health Inference & Forecast Dashboard")
st.write("Upload engine data, select a model, and get predictions along with a 10-15 day forecast.")

# ---------------- Sidebar ---------------- #
st.sidebar.header("⚙ Settings")
model_option = st.sidebar.selectbox("Select Model", ["CNN Only", "LLM Only", "Hybrid (CNN+LLM)"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 10, 15, 12)
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

# ---------------- Main Content ---------------- #
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write("Columns in CSV:")
    st.write(list(data.columns))
    st.dataframe(data.head())

    # ---------------- Metrics Cards ---------------- #
    st.subheader("🔢 Quick Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows in Dataset", data.shape[0])
    col2.metric("Columns in Dataset", data.shape[1])
    col3.metric("Selected Model", model_option)

    # ---------------- Predictions ---------------- #
    st.subheader("🔍 Run Model Predictions")
    if st.button("Run Prediction"):
        if model_option == "CNN Only":
            prediction, sensor_contrib = cnn_predict(data.copy())
        elif model_option == "LLM Only":
            prediction, sensor_contrib = llm_predict(data.copy())
        else:
            prediction, sensor_contrib = hybrid_predict(data.copy())

        st.success("✅ Prediction Completed")
        st.dataframe(prediction)

        st.subheader("📊 Sensor Contribution to Prediction")
        sensor_df = pd.DataFrame(list(sensor_contrib.items()), columns=["Sensor", "Contribution"])
        sensor_df = sensor_df.sort_values(by="Contribution", ascending=False)
        st.bar_chart(sensor_df.set_index("Sensor"))

    # ---------------- Forecast ---------------- #
    st.subheader("📈 Engine Health Forecast")
    if st.button("Generate Forecast"):
        if model_option == "CNN Only":
            forecast = cnn_forecast(data.copy(), forecast_days)
        elif model_option == "LLM Only":
            forecast = llm_forecast(data.copy(), forecast_days)
        else:
            forecast = hybrid_forecast(data.copy(), forecast_days)

        # Display forecast table
        st.dataframe(forecast)

        # Side-by-side metrics for forecast
        total_fail = forecast['Status'].value_counts().get('Fail', 0)
        total_healthy = forecast['Status'].value_counts().get('Healthy', 0)
        col1, col2 = st.columns(2)
        col1.metric("✅ Healthy Days", total_healthy)
        col2.metric("⚠ Failure Days", total_fail)

        # ---------------- Forecast Plot ---------------- #
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(forecast["Date"], forecast["Health Score"], marker='o', linestyle='-', color='royalblue', label='Health Score')
        for i, row in forecast.iterrows():
            if row["Status"] == "Fail":
                ax.scatter(row["Date"], row["Health Score"], color='red', s=100, marker='X')
                ax.text(row["Date"], row["Health Score"]+0.02, "Fail", color='red', fontsize=9)
        ax.set_title("Engine Health Forecast Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Health Score")
        ax.grid(True)
        st.pyplot(fig)

        # Alert
        if total_fail > 0:
            fail_days = forecast[forecast['Status']=="Fail"]['Date'].dt.strftime('%Y-%m-%d').tolist()
            st.warning(f"⚠ {total_fail} Failure(s) expected on: {', '.join(fail_days)}")
        else:
            st.success(f"✅ Engine stable. No failures in next {forecast_days} days.")

else:
    st.info("Upload a CSV file from sidebar to start analysis.")
