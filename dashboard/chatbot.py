from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def explain_churn(customer_data, churn_probability):

    prompt = f"""
    You are a telecom customer retention expert.

    Customer Details:
    {customer_data}

    Churn Probability:
    {churn_probability:.2%}

    Explain:
    1. Why customer may churn
    2. Main risk factors
    3. Retention recommendations

    Keep answer concise and professional.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text