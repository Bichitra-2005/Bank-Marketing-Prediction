import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the Model Assets
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('bank_model_new.pkl')
        s = joblib.load('scaler.pkl')
        c = joblib.load('model_columns.pkl')
        return m, s, c
    except Exception as e:
        return None, None, None

model, scaler, model_columns = load_assets()

# Page Setup
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide", page_icon="🏦")

# Header
st.title("🏦 Bank Marketing Subscription Predictor")
st.markdown("""
Enter the client's details below to predict the likelihood of them subscribing to a term deposit.
*This model does not use 'call duration' to ensure fair, predictive results.*
""")

if model is None:
    st.error("⚠️ Model files not found! Please run 'python train_model.py' first to generate the brain.")
    st.stop()

# 2. Input UI
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("👤 Client Demographics")
    # Using step=1 and value as int makes typing easier
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    
    # Text input for balance if you find number_input clunky to type in
    balance = st.number_input("Annual Balance ($)", value=1000, step=100, help="Total savings in account")
    
    job = st.selectbox("Job Type", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    
    education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])

with col2:
    st.subheader("📊 Campaign & History")
    
    housing = st.radio("Has Housing Loan?", ['yes', 'no'], horizontal=True)
    
    loan = st.radio("Has Personal Loan?", ['yes', 'no'], horizontal=True)
    
    contact = st.selectbox("Contact Communication", ['cellular', 'telephone', 'unknown'])
    
    month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    
    poutcome = st.selectbox("Previous Campaign Outcome", ['success', 'failure', 'other', 'unknown'])

st.divider()

# 3. Prediction Logic
if st.button("Analyze Subscription Probability", type="primary", use_container_width=True):
    
    # Create empty DF with zeros
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill Numerical Values
    input_df.at[0, 'age'] = int(age)
    input_df.at[0, 'balance'] = int(balance)
    input_df.at[0, 'day'] = 15  # Median day
    input_df.at[0, 'campaign'] = 1
    input_df.at[0, 'pdays'] = -1
    input_df.at[0, 'previous'] = 0
    
    # Fill Categorical Values
    # We use a loop to handle the prefixes (job_management, marital_married, etc.)
    categorical_mappings = {
        'job': job, 
        'marital': marital, 
        'education': education, 
        'contact': contact, 
        'month': month,
        'poutcome': poutcome
    }
    
    for prefix, value in categorical_mappings.items():
        col_name = f"{prefix}_{value}"
        if col_name in model_columns:
            input_df.at[0, col_name] = 1
            
    # Binary Yes/No columns
    if housing == 'yes': input_df.at[0, 'housing_yes'] = 1
    if loan == 'yes': input_df.at[0, 'loan_yes'] = 1

    # Apply Scaling
    input_scaled = scaler.transform(input_df)
    
    # Calculate Results
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    yes_prob = probabilities[1]
    
    # Display Results
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.success("### 🎯 Client is LIKELY to Subscribe")
            st.write("Client is **likely** to subscribe.")
        else:
            st.error("### ❌ Client is UNLIKELY to Subscribe")
            st.write("Client is **not likely** to subscribe.")

    with res_col2:
        st.write(f"**Subscription Probability: {yes_prob:.1%}**")
        st.progress(yes_prob)
        
        # Adding a helpful tip based on the probability
        if yes_prob > 0.7:
            st.info("💡 Tip: This is a high-priority lead. Follow up immediately!")
        elif yes_prob < 0.3:
            st.warning("💡 Tip: Financial indicators (balance/loans) suggest low interest.")