# Fraud Detection Demo with XGBoost

A Streamlit app that demonstrates a trained XGBoost fraud detection model.

## Features
- Live simulation of random transactions with real-time fraud predictions
- Upload your own CSV of transactions for batch prediction
- Adds `fraud_proba` (probability) and `is_fraud` (0/1 prediction) columns on uploaded csv files
- st.warning("""
**Important: Your CSV must contain these exact columns:**
- customer_id
- trans_ts (e.g., 2025-01-01 12:30:00)
- amount_usd
- dist_to_home_km
- card_present (0 or 1)
- ip_country
- device_os
- merchant_category
- device_lang
- ip_isp

Extra columns are fine. Do not include prediction columns like is_fraud.
Sort by customer_id and trans_ts (oldest first) for accurate velocity features.
""")
