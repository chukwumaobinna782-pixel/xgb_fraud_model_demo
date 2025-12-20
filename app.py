# app.py
import streamlit as st
import pandas as pd
pd.set_option('styler.render.max_elements', 1000000)
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
        model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
        model.load_model('fraud_detection_model.json')
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
# Preprocessing Function
# -------------------------------
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
    X = df[features]
    return X, df

# -------------------------------
# Prediction Function (Clean - no SHAP)
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
# Decision Legend
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
            <div style="background-color: {bg}; color: {color}; padding: 12px; border-radius: 12px; font-weight: bold; text-align: center;">
            {action}<br><small>{score_range}</small>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------
# Clean Result Display (used in BOTH modes - no explanations)
# -------------------------------
def display_results(df_result):
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", len(df_result))
    actions = ["Auto-Approve", "Low Risk - Monitor", "Manual Review", "High Priority Review", "Auto-Decline"]
    for i, action in enumerate(actions):
        count = (df_result['decision'] == action).sum()
        short_name = action.split(" - ")[0] if " - " in action else action
        col = [col2, col3, col4, col5, col5][i]
        col.metric(short_name, count)

    # Legend
    show_decision_legend()

    # Styled table
    display_cols = ['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country',
                    'risk_score', 'fraud_probability', 'decision']
    display_df = df_result[display_cols].copy()
    display_df['trans_ts'] = display_df['trans_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['fraud_probability'] = display_df['fraud_probability'].map('{:.2%}'.format)
    display_df = display_df.rename(columns={'fraud_probability': 'Fraud Prob'})

    styled = display_df.style.applymap(highlight_action, subset=['decision'])

    st.subheader("Transaction Decisions")
    st.dataframe(styled, use_container_width=True)

    # Download button
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results with Predictions",
        data=csv,
        file_name="fraud_detection_results.csv",
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
# Modes
# -------------------------------
# -------------------------------
# Modes
# -------------------------------
if mode == "Live Demo (Simulate Transactions)":
    st.header("🔴 Live Streaming Simulation Mode")
    st.write("Watch transactions arrive in real-time (3–5 per second), get instantly scored by the model, "
             "and see live decisions + SHAP explanations for the latest transaction — just like a production fraud monitoring dashboard!")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        tx_per_second = st.slider("Transactions per second", 1, 10, 4, help="Higher = faster stream")
    with col2:
        max_transactions = st.number_input("Auto-stop after (transactions)", min_value=50, max_value=10000, value=1000, step=50)
    with col3:
        st.markdown("")  # Spacer
        st.markdown("**Status:** Ready to stream")

    delay_between_tx = 1.0 / tx_per_second

    # Initialize session state
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
        st.session_state.historical_raw = pd.DataFrame()  # Raw input transactions (growing)
        st.session_state.results = pd.DataFrame()        # Full results with predictions
        st.session_state.generated_count = 0

    # Cached SHAP explainer (load once)
    @st.cache_resource
    def get_shap_explainer():
        import shap
        return shap.TreeExplainer(model)

    explainer = get_shap_explainer()

    # Start / Stop button
    if st.session_state.streaming_active:
        if st.button("🛑 Stop Streaming Simulation"):
            st.session_state.streaming_active = False
            st.rerun()
    else:
        if st.button("▶️ Start Live Stream"):
            st.session_state.streaming_active = True
            st.session_state.historical_raw = pd.DataFrame()
            st.session_state.results = pd.DataFrame()
            st.session_state.generated_count = 0
            st.rerun()

    # Placeholders for live updates
    placeholder_metrics = st.empty()
    placeholder_legend = st.empty()
    placeholder_table = st.empty()
    placeholder_latest_explanation = st.empty()
    placeholder_download = st.empty()

    # Streaming loop
    if st.session_state.streaming_active:
        # Generate one new transaction
        customer_id = np.random.choice([f"cust_{i:04d}" for i in range(1, 21)])
        trans_time = datetime.now()

        new_tx = pd.DataFrame({
            'customer_id': [customer_id],
            'trans_ts': [trans_time],
            'amount_usd': [round(np.random.lognormal(4.5, 1.0), 2)],
            'dist_to_home_km': [round(np.random.exponential(500).clip(0, 10000), 1)],
            'card_present': [np.random.choice([0, 1], p=[0.3, 0.7])],
            'ip_country': [np.random.choice(['US', 'CA', 'GB', 'FR', 'DE', 'RU', 'CN', 'MX'],
                                           p=[0.65, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])],
            'device_os': [np.random.choice(['Windows', 'macOS', 'iOS', 'Android', 'Linux'],
                                          p=[0.35, 0.25, 0.2, 0.15, 0.05])],
            'merchant_category': [np.random.choice(['Grocery', 'Electronics', 'Travel', 'Clothing', 'Entertainment', 'Dining'])],
            'device_lang': [np.random.choice(['en-US', 'en-GB', 'es-ES', 'fr-FR', 'ru-RU'],
                                            p=[0.7, 0.1, 0.1, 0.05, 0.05])],
            'ip_isp': [np.random.choice(['Comcast', 'Verizon', 'AT&T', 'Spectrum', 'Unknown'])],
        })

        # Append to growing historical data
        st.session_state.historical_raw = pd.concat(
            [st.session_state.historical_raw, new_tx], ignore_index=True
        )

        # Predict on full history (ensures correct customer-level features)
        df_result = predict_fraud(st.session_state.historical_raw.copy())

        # Store full results
        st.session_state.results = df_result
        st.session_state.generated_count += 1

        # Auto-stop
        if st.session_state.generated_count >= max_transactions:
            st.session_state.streaming_active = False
            st.success(f"✅ Simulation complete: {max_transactions} transactions processed!")

        # Small delay to control speed
        time.sleep(delay_between_tx)

        # Trigger rerun to update UI
        st.rerun()

    # -------------------------------
    # Live Display (updated every rerun)
    # -------------------------------
    if not st.session_state.results.empty:
        df_live = st.session_state.results

        # --- Metrics ---
        with placeholder_metrics:
            cols = st.columns(6)
            cols[0].metric("Processed", len(df_live))
            actions = ["Auto-Approve", "Low Risk - Monitor", "Manual Review", "High Priority Review", "Auto-Decline"]
            for i, action in enumerate(actions):
                count = (df_live['decision'] == action).sum()
                short = action.split(" - ")[0] if " - " in action else action
                cols[i+1].metric(short, count)

        # --- Legend ---
        with placeholder_legend:
            show_decision_legend()

        # --- Table ---
        with placeholder_table:
            st.subheader("📊 Live Transaction Decisions")
            display_cols = ['customer_id', 'trans_ts', 'amount_usd', 'dist_to_home_km', 'ip_country',
                            'risk_score', 'fraud_probability', 'decision']
            disp = df_live[display_cols].copy()
            disp['trans_ts'] = disp['trans_ts'].dt.strftime('%H:%M:%S')
            disp['fraud_probability'] = disp['fraud_probability'].map('{:.2%}'.format)
            disp = disp.rename(columns={'fraud_probability': 'Fraud Prob'})
            styled = disp.style.map(highlight_action, subset=['decision'])  # Updated: .map instead of deprecated .applymap
            st.dataframe(styled, use_container_width=True, height=600)

        # --- Latest Transaction SHAP Explanation ---
        with placeholder_latest_explanation:
            latest_row = df_live.iloc[-1]
            latest_X, _ = preprocess_df(st.session_state.historical_raw.tail(1))
            shap_values_latest = explainer.shap_values(latest_X)

            def get_top_reasons(shap_vals, X_row, top_n=3):
                contributions = pd.Series(shap_vals[0], index=features).abs().sort_values(ascending=False)
                reasons = []
                for feat in contributions.index[:top_n]:
                    val = X_row[feat].iloc[0]
                    impact = shap_vals[0][features.index(feat)]
                    direction = "increases" if impact > 0 else "decreases"
                    reasons.append(f"• **{feat}** = {val} → {direction} risk by {abs(impact):.2f}")
                return "\n".join(reasons)

            explanation_text = get_top_reasons(shap_values_latest, latest_X, top_n=4)

            st.markdown("### 🔍 Latest Transaction Explanation")
            st.markdown(f"**Customer:** {latest_row['customer_id']} | "
                        f"**Amount:** ${latest_row['amount_usd']:.2f} | "
                        f"**Risk Score:** {latest_row['risk_score']}/999 → **{latest_row['decision']}**")
            st.markdown("**Top Contributing Factors:**")
            st.markdown(explanation_text, unsafe_allow_html=True)

        # --- Download ---
        with placeholder_download:
            csv_live = df_live.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Current Live Results",
                data=csv_live,
                file_name=f"live_fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:  # Upload Your Own CSV (unchanged, with your fixes)
    st.header("📂 Upload Your Own Transaction CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload['trans_ts'] = pd.to_datetime(df_upload['trans_ts'], errors='coerce')
            df_upload['card_present'] = pd.to_numeric(df_upload['card_present'], errors='coerce').fillna(0).astype(int)
            df_upload['amount_usd'] = pd.to_numeric(df_upload['amount_usd'], errors='coerce')
            df_upload['dist_to_home_km'] = pd.to_numeric(df_upload['dist_to_home_km'], errors='coerce')
            if df_upload['trans_ts'].isna().any():
                st.error("Some trans_ts values could not be parsed. Check format (e.g., 2025-01-01 10:00:00).")
            else:
                st.success(f"✅ Loaded {len(df_upload)} transactions.")
                st.dataframe(df_upload.head(10), use_container_width=True)
                if st.button("🔍 Run Fraud Detection"):
                    with st.spinner("Processing and predicting on all transactions..."):
                        df_result = predict_fraud(df_upload)
                    display_results(df_result)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Ensure CSV has all required columns and correct types.")
