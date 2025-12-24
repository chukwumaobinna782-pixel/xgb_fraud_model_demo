# app.py - Enhanced Fraud Detection Demo (December 2025)
# GenAI-augmented + Group Risk Features | Risk Score 0–999

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import shap  # Only used in Live Demo
import random

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Enhanced Fraud Detection Demo", layout="wide")
st.title("🛡️ Enhanced XGBoost Fraud Detection Model Demo")
st.markdown("""
**December 2025 Model** — Real-time fraud risk scoring with explainable decisions using a **0–999 risk score**.

Now powered by:
- GenAI synthetic data (proactive against emerging threats)
- Group risk features (detects fraud rings via ISP, device, merchant patterns)
""")

# -------------------------------
# Load Model & Config
# -------------------------------
@st.cache_resource
def load_model_and_config():
    try:
        model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
        model.load_model('fraud_model_enhanced.json')
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        features = config['features']
        # Use same thresholds as before (you can update config if needed)
        thresholds = {
            "auto_approve": 50,
            "monitor": 200,
            "manual_review": 500,
            "high_priority": 800
        }
        multiplier = 999  # Standard 0–999 scale
        version = config.get('model_version', 'Enhanced v1')
        trained_date = config.get('training_date', '2025-12-24')
        st.sidebar.success(f"✅ {version} loaded")
        st.sidebar.caption(f"Trained: {trained_date}")
        return model, config, features, thresholds, multiplier
    except Exception as e:
        st.error("🚨 Failed to load model or config.")
        st.error(f"Error: {str(e)}")
        st.stop()

model, config, features, thresholds, multiplier = load_model_and_config()

# -------------------------------
# Preprocessing Function (Updated for Group Features)
# -------------------------------
def preprocess_df(df_raw):
    df = df_raw.copy()
    
    # Basic signals
    df['card_present'] = df['card_present'].astype(int)
    df['foreign_ip'] = (df['ip_country'] != 'US').astype(int)
    df['risky_ip_country'] = df['ip_country'].isin(['RU', 'CN', 'KP', 'IR', 'VE']).astype(int)
    df['uncommon_os'] = df['device_os'].isin(['Linux', 'ChromeOS']).astype(int)
    
    # Time features
    df['trans_ts'] = pd.to_datetime(df['trans_ts'])
    df['hour_of_day'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.day_of_week
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Categorical
    cat_cols = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Velocity
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
    
    # Device profile (for group features)
    df['device_profile'] = (df['device_os'].astype(str) + '_' + df['device_lang'].astype(str)).astype('category')
    
    # Group Risk Features
    isp_stats = df.groupby('ip_isp').agg(
        isp_tx_count=('customer_id', 'size'),
        isp_fraud_count=('is_fraud', 'sum') if 'is_fraud' in df.columns else ('is_fraud', lambda x: np.nan),
        isp_unique_customers=('customer_id', 'nunique')
    ).reset_index()
    isp_stats['isp_fraud_rate'] = isp_stats['isp_fraud_count'] / isp_stats['isp_tx_count']
    isp_stats['isp_fraud_rate'] = isp_stats['isp_fraud_rate'].fillna(0)
    df = df.merge(isp_stats[['ip_isp', 'isp_fraud_rate', 'isp_tx_count', 'isp_unique_customers']], on='ip_isp', how='left')
    
    device_stats = df.groupby('device_profile').agg(
        dev_tx_count=('customer_id', 'size'),
        dev_fraud_count=('is_fraud', 'sum') if 'is_fraud' in df.columns else ('is_fraud', lambda x: np.nan),
        dev_unique_customers=('customer_id', 'nunique')
    ).reset_index()
    device_stats['dev_fraud_rate'] = device_stats['dev_fraud_count'] / device_stats['dev_tx_count']
    device_stats['dev_fraud_rate'] = device_stats['dev_fraud_rate'].fillna(0)
    df = df.merge(device_stats[['device_profile', 'dev_fraud_rate', 'dev_tx_count', 'dev_unique_customers']], on='device_profile', how='left')
    
    df_sorted = df.sort_values(['merchant_category', 'trans_ts']).reset_index(drop=True)
    df['merchant_prev_count'] = df_sorted.groupby('merchant_category').cumcount()
    df['merchant_prev_fraud'] = (
        df_sorted.groupby('merchant_category')['is_fraud']
        .transform(lambda x: x.shift(1).cumsum().fillna(0)) if 'is_fraud' in df.columns else 0
    )
    df['merchant_prev_fraud_rate'] = df['merchant_prev_fraud'] / df['merchant_prev_count'].clip(lower=1)
    df['merchant_prev_fraud_rate'] = df['merchant_prev_fraud_rate'].fillna(0)
    
    X = df[features]
    return X, df

# -------------------------------
# Prediction Function
# -------------------------------
def predict_fraud(df_raw):
    X, df_processed = preprocess_df(df_raw)
    prob = model.predict_proba(X)[:, 1]
    risk_score = (prob * multiplier).round().astype(int)
    
    def get_action(score):
        if score <= thresholds['auto_approve']:
            return "Auto-Approve"
        elif score <= thresholds['monitor']:
            return "Low Risk - Monitor"
        elif score <= thresholds['manual_review']:
            return "Manual Review"
        elif score <= thresholds['high_priority']:
            return "High Priority Review"
        else:
            return "Auto-Decline"
    
    actions = [get_action(s) for s in risk_score]
    df_processed['fraud_probability'] = prob
    df_processed['risk_score'] = risk_score
    df_processed['decision'] = actions
    return df_processed

# -------------------------------
# Styling & Legend
# -------------------------------
def highlight_action(val):
    color_map = {
        "Auto-Approve": ("#d4edda", "#155724"),
        "Low Risk - Monitor": ("#e7f5ff", "#0c5460"),
        "Manual Review": ("#fff3cd", "#856404"),
        "High Priority Review": ("#f8d7da", "#721c24"),
        "Auto-Decline": ("#f5c6cb", "#721c24")
    }
    bg, text = color_map.get(val, ("#f8f9fa", "#495057"))
    return f'background-color: {bg}; color: {text}; padding: 10px 16px; border-radius: 12px; font-weight: bold; text-align: center;'

def show_decision_legend():
    st.markdown("### 📊 Risk Score Legend (0–999)")
    cols = st.columns(5)
    labels = [
        ("0–50", "Auto-Approve", "#d4edda", "#155724"),
        ("51–200", "Low Risk - Monitor", "#e7f5ff", "#0c5460"),
        ("201–500", "Manual Review", "#fff3cd", "#856404"),
        ("501–800", "High Priority Review", "#f8d7da", "#721c24"),
        ("801–999", "Auto-Decline", "#f5c6cb", "#721c24")
    ]
    for col, (score_range, action, bg, color) in zip(cols, labels):
        with col:
            st.markdown(f"""
            <div style="background-color: {bg}; color: {color}; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center;">
            {action}<br><small>{score_range}</small>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------
# Display Results
# -------------------------------
def display_results(df_result):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", len(df_result))
    actions = ["Auto-Approve", "Low Risk - Monitor", "Manual Review", "High Priority Review", "Auto-Decline"]
    for i, action in enumerate(actions):
        count = (df_result['decision'] == action).sum()
        short_name = action.split(" - ")[0] if " - " in action else action
        col = [col2, col3, col4, col5, col5][i]
        col.metric(short_name, count)
    
    show_decision_legend()
    
    display_cols = ['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country',
                    'risk_score', 'fraud_probability', 'decision']
    display_df = df_result[display_cols].copy()
    display_df['trans_ts'] = display_df['trans_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['fraud_probability'] = display_df['fraud_probability'].map('{:.2%}'.format)
    display_df = display_df.rename(columns={'fraud_probability': 'Fraud Prob'})
    
    styled = display_df.style.applymap(highlight_action, subset=['decision'])
    st.subheader("Transaction Decisions")
    st.dataframe(styled, use_container_width=True)
    
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results with Predictions",
        data=csv,
        file_name="fraud_detection_results_enhanced.csv",
        mime="text/csv"
    )

# -------------------------------
# Sample CSV Template
# -------------------------------
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

# -------------------------------
# UI
# -------------------------------
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

# -------------------------------
# Live Demo Mode - 10k tx, ~2% fraud, SHAP explanations
# -------------------------------
if mode == "Live Demo (Simulate Transactions)":
    st.header("🔴 Live Simulation Mode")
    st.write("Generate up to **10,000 realistic transactions** (~2% fraud rate) and see instant decisions + explanations.")
    
    num_tx = st.slider("Number of transactions to simulate", 100, 10000, 5000, 500)
    
    if st.button("Generate & Predict Simulated Transactions"):
        with st.spinner(f"Generating {num_tx:,} transactions and running predictions..."):
            # (Same realistic transaction generation as before)
            customer_pool = [f"cust_{i:05d}" for i in range(1, 2001)]
            customer_ids = np.random.choice(customer_pool, num_tx)
            
            base_time = datetime(2025, 12, 24)
            time_deltas = np.cumsum(np.random.exponential(scale=15, size=num_tx))
            trans_ts = [base_time + timedelta(minutes=float(d)) for d in time_deltas]
            
            is_fraud = np.random.choice([0, 1], num_tx, p=[0.98, 0.02])
            
            df_sim = pd.DataFrame({
                'customer_id': customer_ids,
                'trans_ts': trans_ts,
                'amount_usd': np.where(is_fraud == 1,
                                      np.random.lognormal(6.0, 1.2, num_tx),
                                      np.random.lognormal(4.5, 1.0, num_tx)).round(2),
                'dist_to_home_km': np.where(is_fraud == 1,
                                           np.random.exponential(2000, num_tx),
                                           np.random.exponential(100, num_tx)).clip(0, 10000).round(1),
                'card_present': np.where(is_fraud == 1, 0, np.random.choice([0, 1], num_tx, p=[0.3, 0.7])),
                'ip_country': np.where(is_fraud == 1,
                                       np.random.choice(['RU', 'CN', 'IR', 'VE', 'UA'], num_tx),
                                       np.random.choice(['US', 'CA', 'GB', 'FR', 'DE'], num_tx, p=[0.7, 0.1, 0.1, 0.05, 0.05])),
                'device_os': np.where(is_fraud == 1,
                                      np.random.choice(['Linux', 'ChromeOS', 'Android'], num_tx, p=[0.5, 0.3, 0.2]),
                                      np.random.choice(['Windows', 'macOS', 'iOS', 'Android'], num_tx, p=[0.4, 0.3, 0.2, 0.1])),
                'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Clothing', 'Entertainment', 'Dining'], num_tx),
                'device_lang': np.random.choice(['en-US', 'ru-RU', 'zh-CN', 'en-GB', 'es-ES'], num_tx),
                'ip_isp': np.where(is_fraud == 1,
                                   np.random.choice(['Unknown', 'ProxyCorp', 'VPNGate'], num_tx),
                                   np.random.choice(['Comcast', 'Verizon', 'AT&T', 'Spectrum'], num_tx)),
                'is_fraud': is_fraud
            })
            
            # Run prediction
            df_result = predict_fraud(df_sim)
            
            # === SHAP ONLY FOR REVIEW/DECLINE TRANSACTIONS ===
            review_or_decline = df_result['decision'].isin([
                "Manual Review",
                "High Priority Review",
                "Auto-Decline"
            ])
            
            if review_or_decline.any():
                st.info(f"🔍 Generating SHAP explanations for {review_or_decline.sum()} transactions requiring review or decline...")
                
                explainer = shap.TreeExplainer(model)
                X_sim, _ = preprocess_df(df_sim)
                
                # Only compute SHAP on the subset needing explanation
                X_review = X_sim[review_or_decline]
                shap_values_review = explainer.shap_values(X_review)
                
                def get_top_reasons(shap_vals_row, x_row, top_n=3):
                    contrib = pd.Series(shap_vals_row, index=features).abs().sort_values(ascending=False)
                    top_feats = contrib.index[:top_n]
                    reasons = []
                    for feat in top_feats:
                        val = x_row[feat]
                        impact = shap_vals_row[features.index(feat)]
                        direction = "increases" if impact > 0 else "decreases"
                        reasons.append(f"• **{feat}** = {val} → {direction} risk by {abs(impact):.2f}")
                    return "\n".join(reasons)
                
                # Map explanations back to full df (only where needed)
                explanations = [""] * len(df_result)
                review_indices = df_result.index[review_or_decline]
                for idx, orig_idx in enumerate(review_indices):
                    explanations[orig_idx] = get_top_reasons(shap_values_review[idx], X_review.iloc[idx])
                
                df_result['explanation'] = explanations
            else:
                df_result['explanation'] = ""
                st.info("✅ All transactions auto-approved or low risk — no explanations needed.")
            
            # Display main results (same beautiful layout)
            display_results(df_result)
            
            # === Individual explanations ONLY for review/decline ===
            if review_or_decline.any():
                st.markdown("---")
                st.subheader("🔍 SHAP Explanations (Only for Review/Decline Transactions)")
                for i, row in df_result[review_or_decline].iterrows():
                    with st.expander(f"TX {i+1}: {row['customer_id']} | ${row['amount_usd']:.2f} | Score: {row['risk_score']}/999 | {row['decision']}"):
                        st.markdown(f"**Decision:** {row['decision']}")
                        st.markdown("**Top Contributing Factors:**")
                        st.markdown(row['explanation'], unsafe_allow_html=True)
            else:
                st.success("🎉 No transactions require manual review or decline — all low risk!")# -------------------------------
# Upload CSV Mode (No explanations)
# -------------------------------
else:
    st.header("📂 Upload Your Own Transaction CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'], errors='coerce')
            if df_upload['trans_ts'].isna().any():
                st.error("Some trans_ts values could not be parsed. Use format like '2025-01-01 10:00:00'.")
            else:
                st.success(f"✅ Loaded {len(df_upload)} transactions.")
                st.dataframe(df_upload.head(10), use_container_width=True)
                if st.button("🔍 Run Fraud Detection"):
                    with st.spinner("Processing all transactions..."):
                        df_result = predict_fraud(df_upload)
                    display_results(df_result)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
