import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import config
from src.model.data_extraction import load_data
from src.model.data_processing import preprocess_data
from src.model.dataloader import create_dataloader
from src.model.model import SentimentClassifier
from src.model.trainer import train_model
from src.model.evaluate import evaluate_and_plot

from huggingface_hub import hf_hub_download
# ------------------------------------------
# Utility functions
# ------------------------------------------

def get_last_chunk_state(state_file="last_chunk.txt"):
    """Return the last processed chunk index from file."""
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return int(f.read().strip())
    return -1  # means no chunk processed yet

def update_last_chunk_state(chunk_idx, state_file="last_chunk.txt"):
    """Update the state file with the last processed chunk index."""
    with open(state_file, "w") as f:
        f.write(str(chunk_idx))

def dataloader_train_test_val(df):
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # change use_weighted_sampler to false to stop using weighted sampler 
    # data = create_dataloader(
    #     df, tokenizer, max_len=config.MAX_LEN, batch_size=config.BATCH_SIZE, use_weighted_sampler=True
    # )
    data = create_dataloader(
        df, tokenizer, max_len=config.MAX_LEN, batch_size=config.BATCH_SIZE
    )
    return data

# ------------------------------------------
# Main training loop
# ------------------------------------------

def main():
    print("Loading dataset...\n")

    # Load dataset
    data = load_data(config.DATASET_PATH, merge_labels=True)

    # Decide chunk size (e.g., 1%)
    CHUNK_FRAC = 1.0
    chunk_size = int(len(data) * CHUNK_FRAC)
    n_chunks = (len(data) // chunk_size) + int(len(data) % chunk_size != 0)

    # Get last processed chunk
    last_chunk = get_last_chunk_state()
    next_chunk = last_chunk + 1

    if next_chunk >= n_chunks:
        print("✅ All chunks have already been processed.")
        return

    # Select the next chunk
    start_idx = next_chunk * chunk_size
    end_idx = min((next_chunk + 1) * chunk_size, len(data))
    data_chunk = data.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"Using chunk {next_chunk+1}/{n_chunks} with {len(data_chunk)} samples")

    # Split train/val/test
    train_data_raw, test_data_raw = train_test_split(
        data_chunk, test_size=config.TEST_SIZE, random_state=42
    )
    train_data, val_data = preprocess_data(
        train_data_raw, test_size=config.VAL_SIZE, max_length=config.MAX_LEN
    )

    # Dataloaders
    train_data = dataloader_train_test_val(train_data)
    val_data = dataloader_train_test_val(val_data)
    test_data = dataloader_train_test_val(test_data_raw)

    # # Initialize or load model
    # model = SentimentClassifier(n_classes=config.N_CLASSES).to(config.DEVICE)
    # best_model_path = os.path.join(
    #     config.MODEL_TRAINING_OUTPUT_DIR, "best_model.pth"
    # )
    # if os.path.exists(best_model_path):
    #     print(f"🔄 Loading previous model from {best_model_path}")
    #     model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

    # Initialize model
    model = SentimentClassifier(n_classes=config.N_CLASSES).to(config.DEVICE)
    best_model_path = os.path.join(config.MODEL_TRAINING_OUTPUT_DIR, "best_model.pth")
    
    if os.path.exists(best_model_path):
        print(f"🔄 Loading previous model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    
    else:
        try:
            print("🌐 No local model found, trying to download from Hugging Face Hub...")
            hf_model_path = hf_hub_download(
                repo_id="Adelanseur/MLOps-Project",   # 👈 hugging face repo
                filename="best_model.pth",
                local_dir=config.MODEL_TRAINING_OUTPUT_DIR,
                force_download=False
            )
            model.load_state_dict(torch.load(hf_model_path, map_location=config.DEVICE))
            print("✅ Loaded model from Hugging Face Hub")
        except Exception as e:
            print(f"⚠️ Could not load model from Hugging Face Hub: {e}")
            print("➡️ Starting training from scratch.")

    # Train on this chunk
    print("Training model...\n")
    trained_model = train_model(
        model, train_data, val_data, device=config.DEVICE, epochs=config.EPOCHS
    )

    # Save progress
    update_last_chunk_state(next_chunk)

    # Evaluate
    print("Evaluating model...\n")
    sentiment_mapper = (config.SENTIMENT_MAPPING)
    
    evaluate_and_plot(
        trained_model,
        test_data,
        torch.nn.CrossEntropyLoss(),
        config.DEVICE,
        class_names=list(sentiment_mapper.values()),
        run_folder=config.MODEL_EVALUATION_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
