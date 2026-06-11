from chatbot import explain_churn
import streamlit as st
import pandas as pd
import joblib
# Load trained pipeline
pipeline = joblib.load(
    "models/customer_churn_model.pkl"
)

st.set_page_config(
    page_title="Customer Churn Predction",
    page_icon="📊"
)
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to churn."

)
#Customer Inputs
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# --------------------------
# Prediction Button
# --------------------------

if st.button("Predict Churn"):

    customer = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService":[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless_billing],
        "PaymentMethod":[payment_method],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]
    })

    prediction = pipeline.predict(customer)

    probability = pipeline.predict_proba(customer)

    churn_probability = probability[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(
            f"⚠ Customer likely to churn ({churn_probability:.2%})"
        )
    else:
        st.success(
            f"✅ Customer likely to stay ({1-churn_probability:.2%})"
        )

    st.write(
        f"Churn Probability: {churn_probability:.2%}"
    )
    if st.button("🤖 Explain Prediction"):

        with st.spinner("Analyzing customer..."):

           explanation = explain_churn(
               customer.to_dict(),
               churn_probability
        )

           st.markdown(explanation)