import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
import json
from PIL import Image
from datetime import datetime

from src/api import database  

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
def get_latest_output_folder(base_path="/app/outputs/training_evaluation/evaluation"):
    """Return the most recent run folder inside container"""
    if not os.path.exists(base_path):
        return None
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    if not folders:
        return None
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
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Prediction Logs"])

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
    latest_folder = get_latest_output_folder("outputs/training_evaluation/evaluation")
    
    if latest_folder:
        st.subheader(f"Latest Outputs Folder: {latest_folder}")
        
        # Load metrics.txt
        metrics_file = os.path.join(latest_folder, "metrics.txt")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics_content = f.read()
            st.text_area("Metrics.txt", metrics_content, height=200)
        else:
            st.warning("No metrics.txt found in latest folder.")
        
        # Load and display PNG plots
        plot_files = [
            "classification_report.png",
            "confusion_matrix.png",
            "confidence_histogram.png"
        ]
        
        for plot_name in plot_files:
            plot_path = os.path.join(latest_folder, plot_name)
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                st.image(img, caption=plot_name.replace("_", " ").replace(".png", ""), use_container_width=True)
            else:
                st.warning(f"{plot_name} not found in latest folder.")
        
        # Optional: display timestamp of last update
        timestamp = datetime.fromtimestamp(os.path.getmtime(latest_folder))
        st.caption(f"Last updated: {timestamp}")
    else:
        st.warning("No outputs folder found.")

# -----------------------------
# Prediction Logs Tab
# -----------------------------
with tab3:
    st.header("🗃️ Recent Prediction Logs")
    rows = database.fetch_recent_predictions(limit=20)
    if not rows:
        st.warning("No predictions logged yet.")
    else:
        # Convert to DataFrame for nicer display
        df = pd.DataFrame(rows, columns=[
            "timestamp", "input_text", "predicted_label", 
            "confidence", "model_version", "model_type", "latency_ms"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df, use_container_width=True)
