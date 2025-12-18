# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Load the saved model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_fraud_model')
    features = joblib.load('feature_names.joblib')
    return model, features

model, features = load_model()

# Function to preprocess a dataframe (engineer features as in training)
def preprocess_df(df):
    # Assume df has the raw columns: amount_usd, dist_to_home_km, card_present, ip_country, device_os,
    # trans_ts (datetime), merchant_category, device_lang, ip_isp, customer_id
    # Note: For prediction, 'is_fraud' should not be present or ignored

    direct_features = ['amount_usd', 'dist_to_home_km']

    df['card_present'] = df['card_present'].astype(int)
    df['foreign_ip'] = (df['ip_country'] != 'US').astype(int)
    df['risky_ip_country'] = df['ip_country'].isin(['RU', 'CN', 'KP', 'IR', 'VE']).astype(int)
    df['uncommon_os'] = df['device_os'].isin(['Linux', 'ChromeOS']).astype(int)

    signal_features = ['card_present', 'foreign_ip', 'risky_ip_country', 'uncommon_os']

    df['hour_of_day'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.day_of_week
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    time_features = ['hour_of_day', 'day_of_week', 'weekend']

    cat_cols = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']

    for col in cat_cols:
        df[col] = df[col].astype('category')

    encoded_cat_features = cat_cols

    df = df.sort_values(['customer_id', 'trans_ts']).reset_index(drop=True)

    df['cust_prev_tx'] = df.groupby('customer_id').cumcount()
    df['time_since_prev_tx_sec'] = df.groupby('customer_id')['trans_ts'].diff().dt.total_seconds().fillna(1e9)
    df['very_quick_succession'] = (df['time_since_prev_tx_sec'] < 300).astype(int)
    df['cust_hist_avg_amount'] = (
        df.groupby('customer_id')['amount_usd']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .fillna(df['amount_usd'].mean())
    )
    df['amount_to_hist_avg_ratio'] = df['amount_usd'] / (df['cust_hist_avg_amount'] + 1)

    velocity_features = ['cust_prev_tx', 'time_since_prev_tx_sec', 'very_quick_succession', 'amount_to_hist_avg_ratio']

    all_features = (
        direct_features + signal_features + time_features + encoded_cat_features + velocity_features
    )

    X = df[all_features]
    return X, df

# Function to predict on a dataframe
def predict_fraud(df):
    X, df_processed = preprocess_df(df)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    df_processed['fraud_proba'] = y_prob
    df_processed['is_fraud'] = y_pred
    return df_processed

# Function to generate simulated transactions
def generate_simulated_transactions(num_tx=10):
    customer_ids = np.random.choice(['cust_1', 'cust_2', 'cust_3'], num_tx)
    base_time = datetime.now() - timedelta(days=30)
    trans_ts = [base_time + timedelta(seconds=np.random.randint(0, 2592000)) for _ in range(num_tx)]  # Random times in last 30 days

    df_sim = pd.DataFrame({
        'customer_id': customer_ids,
        'trans_ts': trans_ts,
        'amount_usd': np.random.uniform(1, 1000, num_tx).round(2),
        'dist_to_home_km': np.random.uniform(0, 5000, num_tx).round(1),
        'card_present': np.random.choice([0, 1], num_tx, p=[0.2, 0.8]),
        'ip_country': np.random.choice(['US', 'CA', 'MX', 'RU', 'CN', 'GB', 'FR'], num_tx, p=[0.7, 0.05, 0.05, 0.025, 0.025, 0.075, 0.075]),
        'device_os': np.random.choice(['Windows', 'macOS', 'iOS', 'Android', 'Linux'], num_tx, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
        'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Clothing', 'Entertainment'], num_tx),
        'device_lang': np.random.choice(['en-US', 'es-MX', 'fr-FR', 'ru-RU'], num_tx, p=[0.8, 0.1, 0.05, 0.05]),
        'ip_isp': np.random.choice(['Comcast', 'Verizon', 'AT&T', 'Unknown'], num_tx),
    })
    df_sim['trans_ts'] = pd.to_datetime(df_sim['trans_ts'])
    return df_sim

# Streamlit app
st.title("Fraud Detection Model Demo")

st.sidebar.header("Options")
mode = st.sidebar.radio("Select Mode", ["Live Demo (Simulate Transactions)", "Upload CSV for Batch Prediction"])

if mode == "Live Demo (Simulate Transactions)":
    st.header("Simulate Transactions")
    num_sim = st.number_input("Number of transactions to simulate", min_value=1, max_value=100, value=10)
    if st.button("Generate and Predict"):
        df_sim = generate_simulated_transactions(num_sim)
        st.subheader("Simulated Raw Transactions")
        st.dataframe(df_sim)

        df_pred = predict_fraud(df_sim)
        st.subheader("Predictions")
        st.dataframe(df_pred[['customer_id', 'trans_ts', 'amount_usd', 'fraud_proba', 'is_fraud']])

        # Show effectiveness metrics (simulated, since no true labels)
        fraud_rate = df_pred['is_fraud'].mean() * 100
        st.write(f"Detected Fraud Rate: {fraud_rate:.2f}%")
        st.write("This demo shows how the model flags potential fraud in simulated data. In real scenarios, compare with actual labels.")

        # Download predicted CSV
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted CSV", csv, "simulated_predictions.csv", "text/csv")

elif mode == "Upload CSV for Batch Prediction":
    st.header("Upload Transaction CSV")
    st.write("Upload a CSV with columns: customer_id, trans_ts (datetime format), amount_usd, dist_to_home_km, card_present (0/1), ip_country, device_os, merchant_category, device_lang, ip_isp")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'])

        st.subheader("Uploaded Data Preview")
        st.dataframe(df_upload.head())

        if st.button("Run Predictions"):
            df_pred = predict_fraud(df_upload)
            st.subheader("Predictions")
            st.dataframe(df_pred)

            # Download updated CSV
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button("Download Updated CSV with Predictions", csv, "predictions.csv", "text/csv")
