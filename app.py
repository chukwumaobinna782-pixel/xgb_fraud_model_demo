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

# Load model safely
@st.cache_resource
def load_model():
    features = joblib.load('feature_names.joblib')
    model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
    model.load_model('xgboost_fraud_model.json')
    return model, features

try:
    model, features = load_model()
except Exception as e:
    st.error("🚨 Failed to load the fraud detection model.")
    st.error(f"Error: {str(e)}")
    st.stop()

# Preprocessing function
def preprocess_df(df_raw):
    df = df_raw.copy()
    
    df['card_present'] = df['card_present'].astype(int)
    df['foreign_ip'] = (df['ip_country'] != 'US').astype(int)
    df['risky_ip_country'] = df['ip_country'].isin(['RU', 'CN', 'KP', 'IR', 'VE']).astype(int)
    df['uncommon_os'] = df['device_os'].isin(['Linux', 'ChromeOS']).astype(int)

    df['hour_of_day'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.day_of_week
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    cat_cols = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

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

    all_features = [
        'amount_usd', 'dist_to_home_km',
        'card_present', 'foreign_ip', 'risky_ip_country', 'uncommon_os',
        'hour_of_day', 'day_of_week', 'weekend',
        'merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp',
        'cust_prev_tx', 'time_since_prev_tx_sec', 'very_quick_succession', 'amount_to_hist_avg_ratio'
    ]

    X = df[all_features]
    return X, df

# Prediction function with correct decision logic
def predict_fraud(df_raw):
    X, df_processed = preprocess_df(df_raw)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    df_processed['fraud_proba'] = y_prob
    df_processed['is_fraud'] = y_pred
    
        # Decision logic - strictly by percentage ranges
        # Ultimate fix: Round probability to 4 decimals FIRST, then make decision on rounded value
    df_processed['fraud_proba_rounded'] = df_processed['fraud_proba'].round(4)  # Matches display rounding
    
    def get_decision(rounded_prob):
        if rounded_prob <= 0.10:
            return "Approved"
        elif 0.11 <= rounded_prob <= 0.49:
            return "Review"
        else:
            return "Flagged"
    
    df_processed['decision'] = df_processed['fraud_proba_rounded'].apply(get_decision)
    return df_processed

# Sample CSV template
def get_sample_csv():
    sample_data = pd.DataFrame({
        'customer_id': ['cust_001', 'cust_001', 'cust_002', 'cust_002', 'cust_003'],
        'trans_ts': ['2025-01-01 10:00:00', '2025-01-01 10:15:00', '2025-01-02 14:30:00', '2025-01-02 14:35:00', '2025-01-03 09:20:00'],
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

# Decision badge styling
def highlight_decision(val):
    if val == "Approved":
        bg = "#d4edda"
        color = "#155724"
    elif val == "Review":
        bg = "#fff3cd"
        color = "#856404"
    else:  # Flagged
        bg = "#f8d7da"
        color = "#721c24"
    return f'background-color: {bg}; color: {color}; padding: 8px 12px; border-radius: 12px; font-weight: bold; text-align: center;'

# Decision Legend (reusable)
def show_decision_legend():
    st.markdown("### 📊 Decision Legend")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #d4edda; color: #155724; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center; border: 1px solid #c3e6cb;">
        ✅ Approved<br><small>Fraud Probability: 0.00% – 0.10%</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background-color: #fff3cd; color: #856404; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center; border: 1px solid #ffeaa7;">
        ⚠️ Review<br><small>Fraud Probability: 0.11% – 0.49%</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background-color: #f8d7da; color: #721c24; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center; border: 1px solid #f5c6cb;">
        🔴 Flagged<br><small>Fraud Probability: ≥ 0.50%</small>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# UI
st.info("""
**Required Columns for CSV Upload:**  
`customer_id`, `trans_ts`, `amount_usd`, `dist_to_home_km`, `card_present`,  
`ip_country`, `device_os`, `merchant_category`, `device_lang`, `ip_isp`
""")

st.download_button(
    label="📥 Download Sample CSV Template",
    data=get_sample_csv().to_csv(index=False).encode('utf-8'),
    file_name="sample_transactions_template.csv",
    mime="text/csv"
)

st.markdown("---")

st.sidebar.header("Choose Mode")
mode = st.sidebar.radio("Select a mode", ["Live Demo (Simulate Transactions)", "Upload Your Own CSV"])

if mode == "Live Demo (Simulate Transactions)":
    st.header("🔴 Live Simulation Mode")
    st.write("Generate realistic transactions and see instant fraud decisions.")

    num_tx = st.slider("Number of transactions to simulate", 5, 100, 20, 5)

    if st.button("Generate & Predict Simulated Transactions"):
        with st.spinner("Generating and predicting..."):
            customer_ids = np.random.choice([f"cust_{i:04d}" for i in range(1, 21)], num_tx)
            base_time = datetime(2025, 1, 1)
            trans_ts = sorted(base_time + timedelta(minutes=np.random.exponential(30)) for _ in range(num_tx))

            df_sim = pd.DataFrame({
                'customer_id': customer_ids,
                'trans_ts': trans_ts,
                'amount_usd': np.random.lognormal(4.5, 1.0, num_tx).round(2),
                'dist_to_home_km': np.random.exponential(500, num_tx).clip(0, 10000).round(1),
                'card_present': np.random.choice([0, 1], num_tx, p=[0.3, 0.7]),
                'ip_country': np.random.choice(['US', 'CA', 'GB', 'FR', 'DE', 'RU', 'CN', 'MX'], num_tx, p=[0.65, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01]),
                'device_os': np.random.choice(['Windows', 'macOS', 'iOS', 'Android', 'Linux'], num_tx, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
                'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Clothing', 'Entertainment', 'Dining'], num_tx),
                'device_lang': np.random.choice(['en-US', 'en-GB', 'es-ES', 'fr-FR', 'ru-RU'], num_tx, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                'ip_isp': np.random.choice(['Comcast', 'Verizon', 'AT&T', 'Spectrum', 'Unknown'], num_tx),
            })
            df_sim['trans_ts'] = pd.to_datetime(df_sim['trans_ts'])

            df_result = predict_fraud(df_sim)

            # Metrics
            approved_count = (df_result['decision'] == 'Approved').sum()
            review_count = (df_result['decision'] == 'Review').sum()
            flagged_count = (df_result['decision'] == 'Flagged').sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", len(df_result))
            col2.metric("✅ Approved", approved_count)
            col3.metric("⚠️ Review", review_count)
            col4.metric("🔴 Flagged", flagged_count)

            # Legend
            show_decision_legend()

            # Table
            display_df = df_result[['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country', 'fraud_proba', 'decision']].copy()
            display_df['fraud_proba'] = display_df['fraud_proba'].round(4).map('{:.2%}'.format)
            display_df['trans_ts'] = display_df['trans_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.subheader("Transaction Decisions")
            styled_df = display_df.style.applymap(highlight_decision, subset=['decision'])
            st.dataframe(styled_df, use_container_width=True)

            # Download
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Results", csv, "simulated_predictions.csv", "text/csv")

else:  # Upload mode
    st.header("📂 Upload Your Own Transaction CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'], errors='coerce')

            if df_upload['trans_ts'].isna().any():
                st.error("Some trans_ts values could not be parsed as dates. Please check the format.")
            else:
                st.success(f"✅ Loaded {len(df_upload)} transactions.")
                st.dataframe(df_upload.head(10))

                if st.button("🔍 Run Fraud Detection"):
                    with st.spinner("Processing and predicting..."):
                        df_result = predict_fraud(df_upload)

                        approved_count = (df_result['decision'] == 'Approved').sum()
                        review_count = (df_result['decision'] == 'Review').sum()
                        flagged_count = (df_result['decision'] == 'Flagged').sum()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Transactions", len(df_result))
                        col2.metric("✅ Approved", approved_count)
                        col3.metric("⚠️ Review", review_count)
                        col4.metric("🔴 Flagged", flagged_count)

                        # Legend
                        show_decision_legend()

                        # Table
                        display_df = df_result[['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country', 'fraud_proba', 'decision']].copy()
                        display_df['fraud_proba'] = display_df['fraud_proba'].round(4).map('{:.2%}'.format)
                        display_df['trans_ts'] = display_df['trans_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

                        st.subheader("Transaction Decisions")
                        styled_df = display_df.style.applymap(highlight_decision, subset=['decision'])
                        st.dataframe(styled_df, use_container_width=True)

                        # Download
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results with Predictions", csv, "batch_fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Ensure your CSV has all required columns and correct data types.")
