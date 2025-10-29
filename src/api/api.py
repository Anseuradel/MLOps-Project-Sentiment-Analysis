# api.py
import os
import random
import logging
from datetime import datetime
from time import time
from typing import Optional, Literal, Dict
from pydantic import BaseModel
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

# optional HF utilities
from huggingface_hub import hf_hub_download, HfFolder

# local models
from src.model.model import SentimentClassifier, MockSentimentClassifier
# Load db function
from src.api import database




# ------------------------------------------------------------------------------
# Configuration from env / defaults
# ------------------------------------------------------------------------------
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
MODEL_REPO = os.getenv("MODEL_REPO", "Adelanseur/MLOps-Project")  # HF repo (optional)
MODEL_FILE = os.getenv("MODEL_FILE", "best_model.pth")            # file inside HF repo
MODEL_PATH_LOCAL = os.getenv("MODEL_PATH", "outputs/best_model.pth")  # fallback local path
MODEL_NAME = os.getenv("MODEL_NAME", "prajjwal1/bert-tiny")       # base HF model used for architecture
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "5"))
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional, for private repos

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Logging & App setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-api")

app = FastAPI(title="Sentiment Model Serving API")
Instrumentator().instrument(app).expose(app)

prediction_counter = Counter("model_predictions_total", "Total number of predictions made")
error_counter = Counter("http_errors_total", "Total API errors")
model_load_error = Counter("model_load_errors_total", "Total model loading failures")

# ------------------------------------------------------------------------------
# Load model (mock or real)
# ------------------------------------------------------------------------------
model = None
tokenizer = None  # only used for real model path

def try_load_real_model():
    """Attempt to download model file from HF or use local path, then load weights."""
    # 1) attempt HF download if HF repo is set
    model_file_path = None
    logger.info(f"Attempting to download {MODEL_FILE} from HF repo {MODEL_REPO} ...")

            # ---- Clear old Hugging Face cache to ensure fresh model ----
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                logger.info("Clearing old Hugging Face model cache...")
                shutil.rmtree(cache_dir)

            # ---- Handle optional token ----
            if HF_TOKEN:
                HfFolder.save_token(HF_TOKEN)

            # ---- Force download of the latest model version ----
            model_file_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                force_download=True  # ensures you don’t use stale files
            )

            logger.info(f"Downloaded model file to: {model_file_path}")

    except Exception as e:
        logger.warning(f"Could not download from HF: {e}")

    # 2) fallback to local path if HF didn't work
    if model_file_path is None or not os.path.exists(model_file_path):
        if os.path.exists(MODEL_PATH_LOCAL):
            model_file_path = MODEL_PATH_LOCAL
            logger.info(f"Using local model path: {model_file_path}")
        else:
            raise FileNotFoundError(
                f"No model found. Tried HF ({MODEL_REPO}/{MODEL_FILE}) and local path ({MODEL_PATH_LOCAL})."
            )

    # 3) instantiate model architecture and load state dict
    net = SentimentClassifier(n_classes=NUM_CLASSES, model_name=MODEL_NAME)
    state = torch.load(model_file_path, map_location=device)
    # Accept either a state_dict or full model dict; if it's a state_dict, load_state_dict:
    if isinstance(state, dict) and not any(k.startswith("_") for k in state.keys()):
        # likely state_dict
        net.load_state_dict(state)
    else:
        # sometimes people save a dict with keys etc; try to handle common case
        try:
            net.load_state_dict(state.get("model_state_dict", state))
        except Exception:
            # attempt direct load (may fail)
            net.load_state_dict(state)

    net.to(device)
    net.eval()
    logger.info("Real model loaded and ready.")
    return net

# Initialize model at startup
if USE_MOCK:
    logger.info("⚡ USE_MOCK=true -> using MockSentimentClassifier (no HF downloads).")
    model = MockSentimentClassifier(n_classes=NUM_CLASSES)
else:
    try:
        model = try_load_real_model()
    except Exception as e:
        model_load_error.inc()
        logger.exception("Failed to load real model; falling back to mock.")
        # fallback to mock so the API stays up (optional)
        model = MockSentimentClassifier(n_classes=NUM_CLASSES)

# ------------------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: int
    prediction_label: str
    confidence: Optional[float]
    probabilities: Optional[Dict[str, float]]
    model_version: str
    model_type: str
    processing_time_ms: float
    timestamp: datetime
    status: Literal["success", "error"]
    error_details: Optional[str] = None

# ------------------------------------------------------------------------------
# Helper: label mapper
# ------------------------------------------------------------------------------
# If you have textual labels mapping in config, you can read it from env or config
DEFAULT_LABELS = ["Horrible","very negative", "negative", "neutral", "positive", "very positive"]
LABELS = DEFAULT_LABELS[:NUM_CLASSES] if len(DEFAULT_LABELS) >= NUM_CLASSES else [str(i) for i in range(NUM_CLASSES)]

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time()
    try:
        prediction_counter.inc()
        # If using mock: expect model.predict and model.predict_proba
        if isinstance(model, MockSentimentClassifier):
            preds = model.predict([request.text])
            probs = model.predict_proba([request.text])[0]
            pred_class = int(preds[0])
        else:
            # Real model: need tokenizer and forward pass. we try simple tokenization using transformers AutoTokenizer.
            # Lazy import to avoid dependency if not used
            from transformers import AutoTokenizer
            # try to instantiate tokenizer only once
            global tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            toks = tokenizer(
                request.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            # toks = {k: v.to(device) for k, v in toks.items()}
            # keep only the keys our model expects
            toks = {k: v.to(device) for k, v in toks.items() if k in ["input_ids", "attention_mask"]}
            with torch.no_grad():
                logits = model(**toks)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs_t = F.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_class = int(probs_t.argmax())
                probs = probs_t

        label = LABELS[pred_class] if pred_class < len(LABELS) else str(pred_class)
        resp = PredictionResponse(
            text=request.text,
            prediction=pred_class,
            prediction_label=label,
            confidence=float(np.max(probs)),
            probabilities={LABELS[i] if i < len(LABELS) else str(i): float(p) for i, p in enumerate(probs)},
            model_version="1.0.0",
            model_type="MockSentimentClassifier" if isinstance(model, MockSentimentClassifier) else "SentimentClassifier",
            processing_time_ms=(time() - start_time) * 1000,
            timestamp=datetime.utcnow(),
            status="success"
        )
        # Log prediction to SQLite
        try:
            database.insert_prediction(
                input_text=request.text,
                predicted_label=label,
                confidence=float(np.max(probs)),
                model_version=resp.model_version,
                model_type=resp.model_type,
                latency_ms=resp.processing_time_ms
            )
        except Exception as db_err:
            logger.warning(f"Failed to log prediction to DB: {db_err}")
            
        return resp

    except Exception as e:
        logger.exception("Prediction failed")
        error_counter.inc()
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_details": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": (time() - start_time) * 1000,
            },
        )

@app.post("/predict_debug")
async def predict_debug(request: Request):
    raw_body = await request.body()
    return {"raw_body": raw_body.decode(), "headers": dict(request.headers)}

# ------------------------------------------------------------------------------  
# Endpoint: Dynamically reload model (Mock or Real)  
# ------------------------------------------------------------------------------  
class ReloadRequest(BaseModel):
    use_mock: bool

@app.post("/reload_model")
async def reload_model(req: ReloadRequest):
    """
    Dynamically reload either the MockSentimentClassifier or the real model without restarting the container.
    """
    global model, USE_MOCK, tokenizer

    USE_MOCK = req.use_mock
    tokenizer = None  # reset tokenizer if switching models

    try:
        if USE_MOCK:
            model = MockSentimentClassifier(n_classes=NUM_CLASSES)
            msg = "MockSentimentClassifier loaded successfully."
        else:
            model = try_load_real_model()
            msg = "Real SentimentClassifier loaded successfully."

        logger.info(msg)
        return {"status": "success", "message": msg, "use_mock": USE_MOCK}

    except Exception as e:
        model_load_error.inc()
        logger.exception("Failed to reload model dynamically.")
        return {"status": "error", "message": f"Failed to reload model: {str(e)}"}

