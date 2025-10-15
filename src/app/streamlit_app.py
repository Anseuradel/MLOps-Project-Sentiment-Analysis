import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import streamlit as st
import requests

st.title("🧠 Sentiment Analysis Web App")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    try:
        response = requests.post(
            "http://ml-service-fastapi:8000/predict",  # ✅ must match service name
            json={"text": user_input}
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction_label']}")
            st.write(f"Confidence: {result['confidence']:.2f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to FastAPI service: {e}")

