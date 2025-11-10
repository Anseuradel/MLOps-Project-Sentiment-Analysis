import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os, sys
import json
from PIL import Image
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import database
  

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="wide")

# Custom dark theme CSS
st.markdown("""
<style>
    /* General dark theme */
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #00C896;
    }
    .stButton>button {
        background-color: #00C896 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        border: none !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00E6A3 !important;
        transform: scale(1.02);
    }
    textarea {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border: 1px solid #333 !important;
    }
    .css-1d391kg, .css-18e3th9 {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
    }
    textarea::placeholder {
        color: #8C8C8C !important;
        opacity: 0.6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Sentiment Analysis Dashboard")
st.write("Analyze text sentiment using a deployed FastAPI ML model.")

API_URL = "http://ml-service-fastapi:8000/predict"

# Sidebar
# st.sidebar.header("⚙️ Model Settings")
# use_mock = st.sidebar.checkbox("Use Mock Model", value=True)
# st.sidebar.write("Switch to real model by setting USE_MOCK=false in docker-compose.")

# Reload model
use_mock = st.radio("Select model:", ["Mock", "Real"])

if st.button("Reload Model"):
    response = requests.post(
        "http://ml-service-fastapi:8000/reload_model",
        json={"use_mock": use_mock == "Mock"}
    )
    st.write(response.json())

# -----------------------------
# Helper functions
# -----------------------------
def get_latest_output_folder(base_path="/app/outputs/training_evaluation/evaluation"):
     """
    Find the most recent evaluation output folder.
    
    Args:
        base_path (str): Base directory containing evaluation run folders
        
    Returns:
        str or None: Path to the latest folder, or None if no folders exist
    """
    if not os.path.exists(base_path):
        return None
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    if not folders:
        return None
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

def load_metrics(output_folder):
    """
    Load evaluation metrics from JSON file if it exists.
    
    Args:
        output_folder (str): Path to the evaluation output folder
        
    Returns:
        dict or None: Loaded metrics dictionary, or None if file doesn't exist
    """
    metrics_file = os.path.join(output_folder, "evaluation.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return json.load(f)
    return None

def load_plot(output_folder, plot_name="accuracy_plot.png"):
    """
    Load a plot image from the output folder.
    
    Args:
        output_folder (str): Path to the evaluation output folder
        plot_name (str): Name of the plot file to load
        
    Returns:
        Image or None: PIL Image object if file exists, None otherwise
    """
    plot_path = os.path.join(output_folder, plot_name)
    if os.path.exists(plot_path):
        return Image.open(plot_path)
    return None


# ------------------------------------------------------
# Sidebar content (Permanent)
# ------------------------------------------------------
st.sidebar.header("💡 About")
st.sidebar.info("""
This interactive dashboard predicts the **sentiment** of user-provided text  
using a fine-tuned **BERT** model deployed with **FastAPI**.

It supports multiple sentiment levels from *very negative* to *very positive*.
""")

st.sidebar.header("🛠️ FAQ / Improvements")
st.sidebar.markdown("""
**Q1:** *Why does it misclassify some neutral reviews?*  
→ The dataset is slightly imbalanced. Adding more balanced neutral samples could help.

**Q2:** *Can inference be faster?*  
→ Yes — deploy on GPU or optimize model weights via distillation.

**Next Steps:**  
- 🔤 Multilingual support  
- 📈 Explainability (SHAP / LIME)  
- 🌐 Online fine-tuning pipeline  
""")

st.sidebar.header("👨‍💻 Credits")
st.sidebar.markdown("""
**Author:** [Adel Anseur](https://github.com/adelanseur)  
**Stack:** Streamlit · FastAPI · PyTorch · Transformers · Docker · Plotly  
""")

# -----------------------------
# Main App Interface
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Prediction Logs"])

# -----------------------------
# Prediction Tab
# -----------------------------
with tab1:
    st.header("Sentiment Prediction")
    user_input = st.text_area(
        "Enter your review:",
        height=150,
        placeholder="e.g. I absolutely love this product! or This was the worst experience ever..."
        )
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

