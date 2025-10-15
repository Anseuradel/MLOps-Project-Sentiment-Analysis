import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

st.title("🧠 Sentiment Analysis Dashboard")
st.write("Analyze text sentiment using a deployed FastAPI ML model.")

API_URL = "http://ml-service-fastapi:8000/predict"

# Sidebar
st.sidebar.header("⚙️ Model Settings")
use_mock = st.sidebar.checkbox("Use Mock Model", value=True)
st.sidebar.write("Switch to real model by setting USE_MOCK=false in docker-compose.")

# Input
st.markdown("### ✍️ Enter your text:")
user_input = st.text_area("Text to analyze", height=150)

# Prediction
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            result = response.json()

            # Show main prediction
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"**Prediction:** {result['prediction_label'].capitalize()}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")
                st.caption(f"Model version: {result['model_version']} | Type: {result['model_type']}")

            # Plot probability distribution
            with col2:
                probs = pd.DataFrame(
                    result["probabilities"].items(),
                    columns=["Label", "Probability"]
                )
                fig = px.bar(
                    probs,
                    x="Label",
                    y="Probability",
                    title="Prediction Probabilities",
                    range_y=[0, 1],
                    text_auto=".2f"
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Error calling prediction API.")

if "history" not in st.session_state:
    st.session_state["history"] = []

# After each successful prediction
st.session_state["history"].append({
    "text": user_input,
    "label": result["prediction_label"],
    "confidence": result["confidence"]
})

if st.session_state["history"]:
    st.markdown("### 🕓 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["history"]))
