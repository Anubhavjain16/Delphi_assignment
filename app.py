import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Solar Site Diagnostic App", layout="wide")
st.title("⚡ Solar Site Anomaly Detection & Forecast Dashboard")

st.markdown("""
This dashboard detects anomalies in Power Factor (PF_avg) using a pre-trained Random Forest model, and forecasts future PF_avg values using a time-series regression model.
""")

# Load trained classifier model
@st.cache_resource
def load_model():
    return joblib.load("best_pf_anomaly_model_RandomForest.joblib")

clf = load_model()

# Simulate and generate data
def generate_data(n_samples=300):
    np.random.seed(42)
    pf = np.random.normal(loc=1.0, scale=0.01, size=n_samples)
    var = np.random.normal(loc=0.0, scale=20, size=n_samples)
    pf[::30] += 0.1  # Inject some anomalies
    df = pd.DataFrame({
        "timestamp": pd.date_range(start=datetime.now(), periods=n_samples, freq="H"),
        "PF_avg": pf,
        "VAR_avg": var
    })
    return df

df_metrics = generate_data()

# Predict using trained model
X = df_metrics[['PF_avg', 'VAR_avg']]
df_metrics['is_anomaly'] = clf.predict(X)

# Forecast future PF_avg values using Linear Regression
def forecast_pf(data, steps=24):
    data = data.copy()
    data['hour'] = np.arange(len(data))
    model = LinearRegression()
    model.fit(data[['hour']], data['PF_avg'])
    future_hours = np.arange(len(data), len(data) + steps)
    future_pf = model.predict(future_hours.reshape(-1, 1))
    future_dates = [data['timestamp'].iloc[-1] + timedelta(hours=i + 1) for i in range(steps)]
    return pd.DataFrame({"timestamp": future_dates, "forecast_pf": future_pf})

forecast_df = forecast_pf(df_metrics)

# Plot anomalies
st.subheader("📉 Detected Anomalies in Power Factor")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_metrics['timestamp'], df_metrics['PF_avg'], label='PF_avg', color='blue', alpha=0.6)
ax1.scatter(df_metrics[df_metrics['is_anomaly'] == 1]['timestamp'],
            df_metrics[df_metrics['is_anomaly'] == 1]['PF_avg'],
            color='red', label='Anomaly', s=20)
ax1.axhline(0.95, color='orange', linestyle='--', linewidth=1)
ax1.axhline(1.05, color='orange', linestyle='--', linewidth=1)
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("PF_avg")
ax1.set_title("Detected Anomalies in Power Factor")
ax1.legend()
st.pyplot(fig1)

# Plot forecast
st.subheader("🔮 Forecast of PF_avg for Next 24 Hours")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df_metrics['timestamp'], df_metrics['PF_avg'], label='Historical PF_avg', alpha=0.5)
ax2.plot(forecast_df['timestamp'], forecast_df['forecast_pf'], label='Forecast PF_avg', linestyle='--', color='green')
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("PF_avg")
ax2.set_title("24-Hour PF_avg Forecast")
ax2.legend()
st.pyplot(fig2)

# Snapshot Table
st.subheader("🧾 Anomaly Snapshot")
st.dataframe(df_metrics[df_metrics['is_anomaly'] == 1][['timestamp', 'PF_avg', 'VAR_avg']].head(20))
