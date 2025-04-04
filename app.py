import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Customer Churn Prediction")
st.markdown("""
This application predicts whether a customer is likely to churn based on their usage patterns.
Enter the customer's details below to get a prediction.
""")

# Generate synthetic data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    monthly_usage = np.random.normal(500, 200, n_samples)
    contract_length = np.random.randint(1, 24, n_samples)
    customer_age = np.random.randint(1, 5, n_samples)  # Years as customer
    support_calls = np.random.poisson(2, n_samples)
    monthly_spend = np.random.normal(100, 30, n_samples)
    
    # Generate target (churn probability)
    churn_prob = (
        0.1 * (monthly_usage < 300) +
        0.2 * (contract_length < 6) +
        0.3 * (customer_age < 2) +
        0.2 * (support_calls > 3) +
        0.2 * (monthly_spend < 80)
    )
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Monthly_Usage': monthly_usage,
        'Contract_Length': contract_length,
        'Customer_Age': customer_age,
        'Support_Calls': support_calls,
        'Monthly_Spend': monthly_spend,
        'Churn': churn
    })
    
    return data

# Train model
@st.cache_data
def train_model(data):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

# Load data and train model
data = generate_data()
model, scaler = train_model(data)

# Sidebar for user input
st.sidebar.header("Customer Details")
monthly_usage = st.sidebar.slider("Monthly Usage (minutes)", 0, 1000, 500)
contract_length = st.sidebar.slider("Contract Length (months)", 1, 24, 12)
customer_age = st.sidebar.slider("Customer Age (years)", 1, 5, 2)
support_calls = st.sidebar.slider("Support Calls (last month)", 0, 10, 2)
monthly_spend = st.sidebar.slider("Monthly Spend ($)", 0, 200, 100)

# Create input data
input_data = pd.DataFrame({
    'Monthly_Usage': [monthly_usage],
    'Contract_Length': [contract_length],
    'Customer_Age': [customer_age],
    'Support_Calls': [support_calls],
    'Monthly_Spend': [monthly_spend]
})

# Make prediction
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display results
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

with col1:
    st.metric("Churn Probability", f"{probability:.1%}")
    
with col2:
    st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Not Churn")

# Visualizations
st.subheader("Data Analysis")

# Churn distribution
st.bar_chart(data['Churn'].value_counts())

# Feature importance
coef = pd.DataFrame({
    'Feature': data.drop('Churn', axis=1).columns,
    'Importance': model.coef_[0]
})
st.bar_chart(coef.set_index('Feature'))

# Display data statistics
st.subheader("Dataset Statistics")
st.dataframe(data.describe(), use_container_width=True) 