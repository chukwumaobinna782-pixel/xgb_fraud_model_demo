# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("🛡️ XGBoost Fraud Detection Model Demo")

# Load model and features
@st.cache_resource
def load_model():
    features = joblib.load('feature_names.joblib')
    model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
    model.load_model('xgboost_fraud_model.json')
    return model, features
model, features = load_model()
    
# Preprocessing function (same as training)
def preprocess_df(df_raw):
    df = df_raw.copy()
    
    # Direct features
    df['card_present'] = df['card_present'].astype(int)
    df['foreign_ip'] = (df['ip_country'] != 'US').astype(int)
    df['risky_ip_country'] = df['ip_country'].isin(['RU', 'CN', 'KP', 'IR', 'VE']).astype(int)
    df['uncommon_os'] = df['device_os'].isin(['Linux', 'ChromeOS']).astype(int)

    # Time features
    df['hour_of_day'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.day_of_week
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Categorical columns
    cat_cols = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Sort for velocity features
    df = df.sort_values(['customer_id', 'trans_ts']).reset_index(drop=True)

    # Velocity features
    df['cust_prev_tx'] = df.groupby('customer_id').cumcount()
    df['time_since_prev_tx_sec'] = df.groupby('customer_id')['trans_ts'].diff().dt.total_seconds().fillna(1e9)
    df['very_quick_succession'] = (df['time_since_prev_tx_sec'] < 300).astype(int)
    df['cust_hist_avg_amount'] = (
        df.groupby('customer_id')['amount_usd']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .fillna(df['amount_usd'].mean())
    )
    df['amount_to_hist_avg_ratio'] = df['amount_usd'] / (df['cust_hist_avg_amount'] + 1)

    # Final feature list
    all_features = [
        'amount_usd', 'dist_to_home_km',
        'card_present', 'foreign_ip', 'risky_ip_country', 'uncommon_os',
        'hour_of_day', 'day_of_week', 'weekend',
        'merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp',
        'cust_prev_tx', 'time_since_prev_tx_sec', 'very_quick_succession', 'amount_to_hist_avg_ratio'
    ]

    X = df[all_features]
    return X, df

# Prediction function
def predict_fraud(df_raw):
    X, df_processed = preprocess_df(df_raw)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    df_processed['fraud_proba'] = y_prob.round(4)
    df_processed['is_fraud'] = y_pred
    
    # Sort back to original order if needed (optional)
    return df_processed

# Sample data for download
def get_sample_csv():
    sample_data = pd.DataFrame({
        'customer_id': ['cust_001', 'cust_001', 'cust_002', 'cust_002', 'cust_003'],
        'trans_ts': [
            '2025-01-01 10:00:00',
            '2025-01-01 10:15:00',
            '2025-01-02 14:30:00',
            '2025-01-02 14:35:00',
            '2025-01-03 09:20:00'
        ],
        'amount_usd': [85.50, 1200.00, 45.75, 320.00, 12.99],
        'dist_to_home_km': [5.2, 4500.0, 12.1, 800.5, 2.0],
        'card_present': [1, 0, 1, 0, 1],
        'ip_country': ['US', 'RU', 'US', 'CN', 'US'],
        'device_os': ['iOS', 'Linux', 'Android', 'Windows', 'macOS'],
        'merchant_category': ['Grocery', 'Electronics', 'Clothing', 'Travel', 'Entertainment'],
        'device_lang': ['en-US', 'ru-RU', 'en-US', 'zh-CN', 'en-US'],
        'ip_isp': ['Verizon', 'Unknown', 'AT&T', 'China Telecom', 'Comcast']
    })
    sample_data['trans_ts'] = pd.to_datetime(sample_data['trans_ts'])
    return sample_data

# Sidebar mode selection
st.sidebar.header("Choose Mode")
mode = st.sidebar.radio(
    "Select a mode",
    ["Live Demo (Simulate Transactions)", "Upload Your Own CSV"]
)

# Required columns warning (shown in both modes)
st.info("""
**Required Columns for CSV Upload:**
`customer_id`, `trans_ts`, `amount_usd`, `dist_to_home_km`, `card_present`, 
`ip_country`, `device_os`, `merchant_category`, `device_lang`, `ip_isp`

Use the sample template below to ensure correct format.
""")

# Download sample CSV button (always visible)
st.download_button(
    label="📥 Download Sample CSV Template",
    data=get_sample_csv().to_csv(index=False).encode('utf-8'),
    file_name="sample_transactions_template.csv",
    mime="text/csv",
    help="Download this template, fill with your data, then upload below."
)

st.markdown("---")

if mode == "Live Demo (Simulate Transactions)":
    st.header("🔴 Live Simulation Mode")
    st.write("Generate random realistic transactions and see real-time fraud predictions.")

    num_tx = st.slider("Number of transactions to simulate", min_value=5, max_value=100, value=20, step=5)

    if st.button("Generate & Predict Simulated Transactions"):
        with st.spinner("Generating transactions and running predictions..."):
            # Generate simulated data
            customer_ids = np.random.choice([f"cust_{i:04d}" for i in range(1, 21)], num_tx)
            base_time = datetime(2025, 1, 1)
            trans_ts = [base_time + timedelta(minutes=np.random.exponential(scale=30)) for _ in range(num_tx)]
            trans_ts.sort()  # chronological order

            df_sim = pd.DataFrame({
                'customer_id': customer_ids,
                'trans_ts': trans_ts,
                'amount_usd': np.random.lognormal(mean=4.5, sigma=1.0, size=num_tx).round(2),
                'dist_to_home_km': np.random.exponential(scale=500, size=num_tx).clip(0, 10000).round(1),
                'card_present': np.random.choice([0, 1], size=num_tx, p=[0.3, 0.7]),
                'ip_country': np.random.choice(['US', 'CA', 'GB', 'FR', 'DE', 'RU', 'CN', 'MX'], size=num_tx, p=[0.65, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01]),
                'device_os': np.random.choice(['Windows', 'macOS', 'iOS', 'Android', 'Linux'], size=num_tx, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
                'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Clothing', 'Entertainment', 'Dining'], size=num_tx),
                'device_lang': np.random.choice(['en-US', 'en-GB', 'es-ES', 'fr-FR', 'ru-RU'], size=num_tx, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                'ip_isp': np.random.choice(['Comcast', 'Verizon', 'AT&T', 'Spectrum', 'Unknown'], size=num_tx),
            })
            df_sim['trans_ts'] = pd.to_datetime(df_sim['trans_ts'])

            # Predict
            df_result = predict_fraud(df_sim)

            # Display results
            st.subheader("🛑 Fraud Predictions")
            fraud_count = df_result['is_fraud'].sum()
            fraud_rate = (fraud_count / len(df_result)) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", len(df_result))
            col2.metric("Flagged as Fraud", fraud_count)
            col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            st.dataframe(
                df_result[['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country', 'fraud_proba', 'is_fraud']]
                .sort_values('fraud_proba', ascending=False)
                .style.format({'fraud_proba': '{:.2%}'})
                .background_gradient(subset=['fraud_proba'], cmap='Reds')
            )

            # Download button
            csv_output = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Results (with all columns)",
                csv_output,
                "simulated_fraud_predictions.csv",
                "text/csv"
            )

else:  # Upload CSV mode
    st.header("📂 Upload Your Own Transaction CSV")
    st.write("Upload a CSV with your transaction data to get batch fraud predictions.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Must match the required columns shown above."
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'trans_ts' in df_upload.columns:
                df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'], errors='coerce')

            if df_upload['trans_ts'].isna().any():
                st.error("Some trans_ts values could not be parsed as dates. Check the format.")
            else:
                st.success(f"✅ Successfully loaded {len(df_upload)} transactions.")
                st.dataframe(df_upload.head(10))

                if st.button("🔍 Run Fraud Detection on Uploaded Data"):
                    with st.spinner("Processing and predicting... This may take a moment."):
                        df_result = predict_fraud(df_upload)

                        fraud_count = df_result['is_fraud'].sum()
                        fraud_rate = (fraud_count / len(df_result)) * 100

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Transactions", len(df_result))
                        col2.metric("Flagged as Fraud", fraud_count)
                        col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

                        st.subheader("🛑 Fraud Predictions")
                        st.dataframe(
                            df_result[['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country', 'fraud_proba', 'is_fraud']]
                            .sort_values('fraud_proba', ascending=False)
                            .style.format({'fraud_proba': '{:.2%}'})
                            .background_gradient(subset=['fraud_proba'], cmap='Reds')
                        )

                        # Download results
                        csv_output = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Predictions (with new columns)",
                            csv_output,
                            "my_transaction_fraud_predictions.csv",
                            "text/csv"
                        )

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Make sure your CSV has the required columns and correct data types.")
