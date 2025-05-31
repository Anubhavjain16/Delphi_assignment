import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Solar Site Diagnostic App", layout="wide")
st.title("🔍 AI-Driven Solar Site Diagnostic Tool")

st.markdown("""
This application detects anomalies in Power Factor (PF_avg) based on uploaded solar site data using an unsupervised AI model (Isolation Forest).
""")

uploaded_file = st.file_uploader("Upload Solar Data Excel File", type=["xlsx"])

if uploaded_file:
    try:
        sheet_data = pd.read_excel(uploaded_file, sheet_name=None)

        # Clean and extract timestamp + key metrics
        dataframes = []
        for sheet_name, df in sheet_data.items():
            df.columns = df.iloc[1]  # Use second row as header
            df = df.iloc[3:].reset_index(drop=True)
            df = df.loc[:, ~df.columns.duplicated()]
            df['timestamp'] = pd.to_datetime(df['field'], errors='coerce')
            dataframes.append(df)

        df_all = pd.concat(dataframes, ignore_index=True)
        df_all = df_all.dropna(subset=['timestamp'])

        key_metrics = ['PF_avg', 'VAR_avg', 'Hz_avg', 'PhV_avg']
        df_metrics = df_all[['timestamp'] + key_metrics].copy()
        for col in key_metrics:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
        df_metrics = df_metrics.dropna()

        # Anomaly Detection
        model = IsolationForest(contamination=0.01, random_state=42)
        df_metrics['anomaly_score'] = model.fit_predict(df_metrics[['PF_avg', 'VAR_avg']])
        df_metrics['is_anomaly'] = df_metrics['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

        # Visualization
        st.subheader("📈 PF_avg Time Series with Anomalies")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_metrics['timestamp'], df_metrics['PF_avg'], label='PF_avg', color='blue', alpha=0.6)
        ax.scatter(df_metrics[df_metrics['is_anomaly'] == 1]['timestamp'],
                   df_metrics[df_metrics['is_anomaly'] == 1]['PF_avg'],
                   color='red', label='Anomaly', s=20)
        ax.axhline(0.95, color='orange', linestyle='--', linewidth=1)
        ax.axhline(1.05, color='orange', linestyle='--', linewidth=1)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PF_avg")
        ax.set_title("Detected Anomalies in Power Factor")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔍 Anomaly Summary Table")
        st.dataframe(df_metrics[df_metrics['is_anomaly'] == 1][['timestamp', 'PF_avg', 'VAR_avg']].head(20))

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a valid Excel file to begin analysis.")
