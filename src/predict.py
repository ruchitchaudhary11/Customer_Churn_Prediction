import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load(
    "models/customer_churn_model.pkl"
)

# New customer data
customer = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [12],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [95.0],
    "TotalCharges": [1200.0]
})

# Prediction
prediction = pipeline.predict(customer)

# Probability
probability = pipeline.predict_proba(customer)

print("Prediction:", prediction[0])

print(
    "Churn Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)

# Human-readable output
if prediction[0] == 1:
    print("Customer is likely to churn.")
else:
    print("Customer is likely to stay.")