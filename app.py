# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("🛡️ XGBoost Fraud Detection Model Demo")
st.markdown("Real-time fraud risk scoring with explainable decisions using a 0–999 risk score.")

# -------------------------------
# Load Model & Config
# -------------------------------
@st.cache_resource
def load_model_and_config():
    try:
        # Load model (XGBoost JSON format)
        model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
        model.load_model('fraud_detection_model.json')  # renamed to match training

        # Load human-readable config
        with open('model_config.json', 'r') as f:
            config = json.load(f)

        features = config['features_used']
        thresholds = config['action_thresholds']
        multiplier = config['risk_score_multiplier']
        version = config.get('model_version', 'Unknown')
        trained_date = config.get('trained_date', 'Unknown')

        st.sidebar.success(f"✅ Model {version} loaded")
        st.sidebar.caption(f"Trained: {trained_date.split('T')[0]}")

        return model, config, features, thresholds, multiplier
    except Exception as e:
        st.error("🚨 Failed to load model or config.")
        st.error(f"Error: {str(e)}")
        st.stop()

model, config, features, thresholds, multiplier = load_model_and_config()

# -------------------------------
# SHAP Explainer (cached for speed)
# -------------------------------
import shap

@st.cache_resource
def get_shap_explainer():
    explainer = shap.TreeExplainer(model)
    return explainer

explainer = get_shap_explainer()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_df(df_raw):
    df = df_raw.copy()

    # Basic signals
    df['card_present'] = df['card_present'].astype(int)
    df['foreign_ip'] = (df['ip_country'] != 'US').astype(int)
    df['risky_ip_country'] = df['ip_country'].isin(['RU', 'CN', 'KP', 'IR', 'VE']).astype(int)
    df['uncommon_os'] = df['device_os'].isin(['Linux', 'ChromeOS']).astype(int)

    # Time features
    df['hour_of_day'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.day_of_week
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Categorical columns → category dtype for XGBoost
    cat_cols = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Sort for correct historical features
    df = df.sort_values(['customer_id', 'trans_ts']).reset_index(drop=True)

    # Velocity & history
    df['cust_prev_tx'] = df.groupby('customer_id').cumcount()
    df['time_since_prev_tx_sec'] = df.groupby('customer_id')['trans_ts'].diff().dt.total_seconds().fillna(1e9)
    df['very_quick_succession'] = (df['time_since_prev_tx_sec'] < 300).astype(int)
    df['cust_hist_avg_amount'] = (
        df.groupby('customer_id')['amount_usd']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .fillna(df['amount_usd'].mean())
    )
    df['amount_to_hist_avg_ratio'] = df['amount_usd'] / (df['cust_hist_avg_amount'] + 1)

    # Ensure exact feature order
    X = df[features]
    return X, df

# -------------------------------
# Prediction + Risk Score + Action + SHAP
# -------------------------------
def predict_fraud(df_raw):
    X, df_processed = preprocess_df(df_raw)

    # Probabilities
    prob = model.predict_proba(X)[:, 1]

    # Risk Score (0–999)
    risk_score = (prob * multiplier).round().astype(int)

    # Action based on thresholds from config
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

    # SHAP values for explanations (top 3 features per row)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=features)

    def get_top_reasons(row_idx, top_n=3):
        contributions = shap_df.iloc[row_idx].abs().sort_values(ascending=False)
        top_features = contributions.index[:top_n]
        reasons = []
        for feat in top_features:
            val = X.iloc[row_idx][feat]
            impact = shap_df.iloc[row_idx][feat]
            direction = "increases" if impact > 0 else "decreases"
            reasons.append(f"• **{feat}** = {val} → {direction} risk by {abs(impact):.1f}")
        return "\n".join(reasons)

    top_reasons = [get_top_reasons(i) for i in range(len(df_processed))]

    # Add results
    df_processed['fraud_probability'] = prob
    df_processed['risk_score'] = risk_score
    df_processed['decision'] = actions
    df_processed['explanation'] = top_reasons

    return df_processed

# -------------------------------
# Action Badge Styling
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

# -------------------------------
# Decision Legend (Updated for Risk Score)
# -------------------------------
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
            <div style="background-color: {bg}; color: {color}; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center; border: 1px solid {bg};">
            {action}<br><small>{score_range}</small>
            </div>
            """, unsafe_allow_html=True)

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
# Shared Result Display Logic
# -------------------------------
def display_results(df_result):
       # Color-coded Summary Metrics (matching legend exactly)
    st.markdown("### 📊 Decision Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transactions", len(df_result))

    # Badge-style colored counts
    badge_style = {
        "Auto-Approve": ("#d4edda", "#155724"),
        "Low Risk - Monitor": ("#e7f5ff", "#0c5460"),
        "Manual Review": ("#fff3cd", "#856404"),
        "High Priority Review": ("#f8d7da", "#721c24"),
        "Auto-Decline": ("#f5c6cb", "#721c24")
    }

    actions_list = ["Auto-Approve", "Low Risk - Monitor", "Manual Review", "High Priority Review", "Auto-Decline"]
    cols_list = [col2, col3, col4, col5, col5]  # Auto-Decline shares column with High Priority (or adjust layout)

    for i, action in enumerate(actions_list):
        count = (df_result['decision'] == action).sum()
        short_label = action.replace(" - ", "\n")
        bg_color, text_color = badge_style[action]
        
        if i < 4:
            with cols_list[i]:
                st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; padding: 12px; border-radius: 12px; text-align: center; font-weight: bold; border: 1px solid {bg_color};">
                    {short_label}<br><span style="font-size: 24px;">{count}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col5:
                st.markdown(f"""
                <div style="background-color: {bg_color}; color: {text_color}; padding: 12px; border-radius: 12px; text-align: center; font-weight: bold; border: 1px solid {bg_color}; margin-top: 10px;">
                    {short_label}<br><span style="font-size: 24px;">{count}</span>
                </div>
                """, unsafe_allow_html=True)t)

    show_decision_legend()

    # Display table
    display_cols = ['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country',
                    'risk_score', 'fraud_probability', 'decision']
    display_df = df_result[display_cols].copy()
    display_df['trans_ts'] = display_df['trans_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['fraud_probability'] = display_df['fraud_probability'].map('{:.2%}'.format)
    display_df = display_df.rename(columns={'fraud_probability': 'Fraud Prob'})

    styled = display_df.style.applymap(highlight_action, subset=['decision'])
    st.subheader("Transaction Decisions")
    st.dataframe(styled, use_container_width=True)

    # Expandable explanations
    st.subheader("🔍 Click to View Explanation")
    for i, row in df_result.iterrows():
        with st.expander(f"Transaction {i+1}: {row['customer_id']} | ${row['amount_usd']:.2f} | Risk Score: {row['risk_score']}/999"):
            st.markdown(f"**Decision:** {row['decision']}")
            st.markdown("**Top Contributing Factors:**")
            st.markdown(row['explanation'], unsafe_allow_html=True)

    # Download full results
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Results with Explanations",
        data=csv,
        file_name="fraud_predictions_with_explanations.csv",
        mime="text/csv"
    )

# -------------------------------
# Modes
# -------------------------------
if mode == "Live Demo (Simulate Transactions)":
    st.header("🔴 Live Simulation Mode")
    st.write("Generate realistic transactions and see instant fraud decisions with explanations.")
    num_tx = st.slider("Number of transactions to simulate", 5, 100, 20, 5)

    if st.button("Generate & Predict Simulated Transactions"):
        with st.spinner("Generating and predicting..."):
            # (Your existing simulation code - unchanged)
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
            display_results(df_result)

else:  # Upload mode
    st.header("📂 Upload Your Own Transaction CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'], errors='coerce')
            if df_upload['trans_ts'].isna().any():
                st.error("Some trans_ts values could not be parsed. Check format (e.g., 2025-01-01 10:00:00).")
            else:
                st.success(f"✅ Loaded {len(df_upload)} transactions.")
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("🔍 Run Fraud Detection"):
                    with st.spinner("Processing and predicting..."):
                        df_result = predict_fraud(df_upload)
                        display_results(df_result)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Ensure CSV has all required columns and correct types.")
