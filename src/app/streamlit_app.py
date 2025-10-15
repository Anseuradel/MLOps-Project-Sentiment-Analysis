import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Sentiment Analysis Dashboard")
st.write("Analyze text sentiment using a deployed FastAPI ML model.")

API_URL = "http://ml-service-fastapi:8000/predict"

# Sidebar
st.sidebar.header("⚙️ Model Settings")
use_mock = st.sidebar.checkbox("Use Mock Model", value=True)
st.sidebar.write("Switch to real model by setting USE_MOCK=false in docker-compose.")

# -----------------------------
# Helper functions
# -----------------------------
def get_latest_output_folder(base_path="outputs/training_evaluation/evaluation"):
    """Return the most recent folder inside outputs/training_evaluation/evaluation"""
    if not os.path.exists(base_path):
        return None
    # List all subfolders
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    if not folders:
        return None
    # Return the one with the latest modification time
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

def load_metrics(output_folder):
    """Load evaluation metrics from evaluation.json if exists"""
    metrics_file = os.path.join(output_folder, "evaluation.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return json.load(f)
    return None

def load_plot(output_folder, plot_name="accuracy_plot.png"):
    """Load a plot image"""
    plot_path = os.path.join(output_folder, plot_name)
    if os.path.exists(plot_path):
        return Image.open(plot_path)
    return None

# -----------------------------
# Tabs for UI
# -----------------------------
tab1, tab2 = st.tabs(["Prediction", "Model Info"])

# -----------------------------
# Prediction Tab
# -----------------------------
with tab1:
    st.header("🧠 Sentiment Prediction")
    user_input = st.text_area("Enter your text:", height=150)
    if st.button("Predict"):
        try:
            response = requests.post(
                API_URL,
                json={"text": user_input},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction_label']}")
                st.write(f"Confidence: {result.get('confidence', 0):.2f}")
            else:
                st.error(f"Error calling prediction API: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# -----------------------------
# Model Info Tab
# -----------------------------
with tab2:
    st.header("📊 Model Evaluation & Metrics")
    latest_folder = get_latest_output_folder("outputs")
    
    if latest_folder:
        st.subheader(f"Latest Outputs Folder: {latest_folder}")
        
        # Load metrics
        metrics = load_metrics(latest_folder)
        if metrics:
            st.write("### Evaluation Metrics")
            st.json(metrics)
        else:
            st.warning("No evaluation.json found in latest folder.")
        
        # Load accuracy plot
        plot = load_plot(latest_folder, "accuracy_plot.png")
        if plot:
            st.image(plot, caption="Accuracy Plot", use_column_width=True)
        else:
            st.warning("No accuracy_plot.png found in latest folder.")
        
        # Display timestamp of last update
        timestamp = datetime.fromtimestamp(os.path.getmtime(latest_folder))
        st.caption(f"Last updated: {timestamp}")
    else:
        st.warning("No outputs folder found.")
