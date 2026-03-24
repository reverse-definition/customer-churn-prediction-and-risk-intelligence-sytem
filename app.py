import streamlit as st
import pandas as pd
import pickle

# Function to generate recommendation
def generate_explanation(prob, risk):
    
    if risk == "High Risk":
        return "Customer is likely to churn. Recommend offering discounts or retention plans."
    
    elif risk == "Medium Risk":
        return "Customer shows moderate churn risk. Recommend engagement strategies."
    
    else:
        return "Customer is stable. Maintain current service quality."

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load feature columns (VERY IMPORTANT)
feature_columns = pickle.load(open("features.pkl", "rb"))

st.title("Customer Churn Prediction")
st.subheader("Prediction Summary")

# Inputs
tenure = st.slider("Tenure", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 150, 70)

# Create full input structure with all features
input_data = pd.DataFrame(columns=feature_columns)

# Initialize all values to 0
input_data.loc[0] = 0

# Fill known inputs
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prob = model.predict_proba(input_scaled)[0][1]

# Risk
if prob < 0.3:
    risk = "Low Risk"
elif prob < 0.7:
    risk = "Medium Risk"
else:
    risk = "High Risk"

# Output
st.write(f"Churn Probability: {prob:.2%}")
# Color-coded risk display
if risk == "High Risk":
    st.error(f"Risk Level: {risk}")
elif risk == "Medium Risk":
    st.warning(f"Risk Level: {risk}")
else:
    st.success(f"Risk Level: {risk}")

# Generate recommendation
recommendation = generate_explanation(prob, risk)

# Display recommendation
st.write(f"Recommendation: {recommendation}")