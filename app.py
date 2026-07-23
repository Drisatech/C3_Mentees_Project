import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the models
@st.cache_resource
def load_models():
    rf_model = joblib.load('random_forest_smote_model.pkl')
    xgb_model = joblib.load('xgboost_smote_model.pkl')
    return rf_model, xgb_model

rf, xgb = load_models()

st.title("🛡️ Fraud Detection System")
st.markdown("Enter transaction details below to predict the likelihood of fraud.")

# Create input form
with st.sidebar:
    st.header("Transaction Details")
    model_choice = st.selectbox("Select Model", ["Random Forest (High Recall)", "XGBoost (High Precision)"])
    
    gender = st.selectbox("Gender", [0, 1])
    age = st.number_input("Age (Scaled)", value=0.0)
    state = st.number_input("State ID", min_value=0)
    city = st.number_input("City ID", min_value=0)
    acc_type = st.selectbox("Account Type", [0, 1, 2])
    amount = st.number_input("Transaction Amount (Scaled)", value=0.0)
    trans_type = st.number_input("Transaction Type ID", value=0)
    merchant_cat = st.number_input("Merchant Category ID", value=0)
    balance = st.number_input("Account Balance (Scaled)", value=0.0)
    device = st.number_input("Device ID", value=0)
    dev_type = st.number_input("Device Type ID", value=0)
    desc = st.number_input("Description ID", value=0)
    day = st.number_input("Day (Scaled)", value=0.0)
    month = st.number_input("Month", value=0.0)
    dow = st.number_input("Day of Week (Scaled)", value=0.0)
    hour = st.number_input("Hour (Scaled)", value=0.0)
    minute = st.number_input("Minute (Scaled)", value=0.0)
    second = st.number_input("Second (Scaled)", value=0.0)

# Prepare input for prediction
input_data = pd.DataFrame([[gender, age, state, city, acc_type, amount, trans_type, merchant_cat, 
                            balance, device, dev_type, desc, day, month, dow, hour, minute, second]],
                          columns=['Gender', 'Age', 'State', 'City', 'Account_Type', 'Transaction_Amount',
                                   'Transaction_Type', 'Merchant_Category', 'Account_Balance', 'Transaction_Device',
                                   'Device_Type', 'Transaction_Description', 'Transaction_Day', 'Transaction_Month',
                                   'Transaction_DayOfWeek', 'Transaction_Hour', 'Transaction_Minute', 'Transaction_Second'])

if st.button("Predict Fraud"):
    model = rf if "Random Forest" in model_choice else xgb
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 ALERT: Potential Fraud Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Transaction looks clean. (Probability of Fraud: {probability:.2%})")
