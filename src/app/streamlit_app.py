import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import streamlit as st
import requests

st.title("🧠 Sentiment Analysis Web App")

# Input
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    response = requests.post(
        # "http://localhost:8000/predict",  # Internal FastAPI endpoint
        "http://ml-service-fastapi:8000/predict",  # ✅ must match service name
        json={"text": user_input}
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction_label']}")
        st.write(f"Confidence: {result['confidence']:.2f}")
    else:
        st.error("Error calling prediction API.")
