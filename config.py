import torch
import os

# --------------------------------------------------------------------------
# Define 5-class sentiment mapping
SENTIMENT_MAPPING = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}


# LABEL_MAPPING = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
LABEL_MAPPING = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4,
}

# --------------------------------------------------------------------------
# Model config
# TOKENIZER_NAME = "bert-base-uncased"
# MODEL_NAME = "bert-base-uncased"

MODEL_NAME = "prajjwal1/bert-tiny"
TOKENIZER_NAME = "prajjwal1/bert-tiny"

EPOCHS = 3
N_CLASSES = 6
DROPOUT = 0.3
MAX_LEN = 64
TEST_SIZE = 0.1
VAL_SIZE = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Real Dataset paths
DATASET_PATH = "Dataset/text.txt"

# --------------------------------------------------------------------------
# Model config

# outpus paths
MODEL_TRAINING_OUTPUT_DIR = "outputs/training_evaluation/training"
MODEL_EVALUATION_OUTPUT_DIR = "outputs/training_evaluation/evaluation"

# -------------------------------------------------------------------------
# Test dataset folder path
TEST_DATA_DIR = "dataset/test_datasets"  # folder containing test data files
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Sqlite database path

# Path inside container (matches your docker-compose volume)
DB_DIR = "/app/db"
DB_PATH = os.path.join(DB_DIR, "predictions.db")

# Ensure folder exists inside container
os.makedirs(DB_DIR, exist_ok=True)
