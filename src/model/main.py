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
# from src.model.trainer import train_model
from src.model.trainer import train_model_with_imbalance_handling, evaluate_imbalanced_model
from src.model.evaluate import evaluate_and_plot
from src.EDA.Data_analysis import analyze_imbalance_solutions

from huggingface_hub import hf_hub_download

# ------------------------------------------
# Utility functions
# ------------------------------------------

def get_last_chunk_state(state_file="last_chunk.txt"):
    """
    Read the last processed chunk index from a state file.
    This enables resumable training across multiple chunks of data.
    
    Args:
        state_file (str): Path to the state tracking file
        
    Returns:
        int: Last processed chunk index, or -1 if no chunks processed yet
    """
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return int(f.read().strip())
    return -1  # means no chunk processed yet

def update_last_chunk_state(chunk_idx, state_file="last_chunk.txt"):
    """
    Update the state file with the last processed chunk index.
    
    Args:
        chunk_idx (int): Index of the chunk that was just processed
        state_file (str): Path to the state tracking file
    """
    with open(state_file, "w") as f:
        f.write(str(chunk_idx))

def dataloader_train_test_val(df):
    """
    Create a DataLoader for the given DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing text and label data
        
    Returns:
        DataLoader: PyTorch DataLoader ready for training/evaluation
    """
    # Initialize tokenizer from configuration
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # Note: Option to use weighted sampler for handling class imbalance
    # change use_weighted_sampler to false to stop using weighted sampler 
    # data = create_dataloader(
    #     df, tokenizer, max_len=config.MAX_LEN, batch_size=config.BATCH_SIZE, use_weighted_sampler=True
    # )
    
    # Create DataLoader with standard configuration
    data = create_dataloader(
        df, tokenizer, max_len=config.MAX_LEN, batch_size=config.BATCH_SIZE
    )
    return data

# ------------------------------------------
# Main training loop
# ------------------------------------------

def main():
    """
    Main training pipeline that processes data in chunks, trains the model,
    and evaluates performance. Supports resumable training and model loading
    from both local storage and Hugging Face Hub.
    """
    print("Loading dataset...\n")

    # Load the full dataset from file
    data = load_data(config.DATASET_PATH, merge_labels=True)

    # Configure chunk-based processing for large datasets
    # Process data in chunks to handle memory constraints or enable incremental learning
    CHUNK_FRAC = 0.9  # Fraction of data to use per chunk (90%)
    chunk_size = int(len(data) * CHUNK_FRAC)
    n_chunks = (len(data) // chunk_size) + int(len(data) % chunk_size != 0)

    # Get last processed chunk to enable resumable training
    last_chunk = get_last_chunk_state()
    next_chunk = last_chunk + 1

    # Check if all chunks have been processed
    if next_chunk >= n_chunks:
        print("✅ All chunks have already been processed.")
        return

    # Select the next chunk to process
    start_idx = next_chunk * chunk_size
    end_idx = min((next_chunk + 1) * chunk_size, len(data))
    data_chunk = data.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"Using chunk {next_chunk+1}/{n_chunks} with {len(data_chunk)} samples")

    # Split data into train/test sets
    train_data_raw, test_data_raw = train_test_split(
        data_chunk, test_size=config.TEST_SIZE, random_state=42
    )
    
    # Further split training data into train/validation sets and preprocess
    train_data, val_data = preprocess_data(
        train_data_raw, test_size=config.VAL_SIZE, max_length=config.MAX_LEN
    )

    # Create DataLoaders for training, validation, and testing
    train_data = dataloader_train_test_val(train_data)
    val_data = dataloader_train_test_val(val_data)
    test_data = dataloader_train_test_val(test_data_raw)

    # Model initialization with fallback loading strategy
    model = SentimentClassifier(n_classes=config.N_CLASSES).to(config.DEVICE)
    best_model_path = os.path.join(config.MODEL_TRAINING_OUTPUT_DIR, "best_model.pth")
    
    # Try to load model from local storage first
    if os.path.exists(best_model_path):
        print(f"🔄 Loading previous model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    
    else:
        # If no local model, try to download from Hugging Face Hub
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
            # If both local and HF loading fail, start from scratch
            print(f"⚠️ Could not load model from Hugging Face Hub: {e}")
            print("➡️ Starting training from scratch.")

    # Train the model on the current chunk
    # print("Training model...\n")
    # trained_model = train_model(
    #     model, train_data, val_data, device=config.DEVICE, epochs=config.EPOCHS
    # )

    # NEW: Train with imbalance handling
    print("Training model with imbalance handling...\n")
    trained_model = train_model_with_imbalance_handling(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        df_train=train_data_raw,  # Pass training data for class weight calculation
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,  # Make sure this is in your config
        use_focal_loss=True,      # Enable focal loss
        use_class_weights=True    # Enable class weights
    )

    
    # Update state to track progress (enables resumable training)
    update_last_chunk_state(next_chunk)

    # # Evaluate model performance on test set
    # print("Evaluating model...\n")
    # sentiment_mapper = (config.SENTIMENT_MAPPING)

    # NEW: Evaluate with imbalance-aware metrics
    print("Evaluating model with imbalance-aware metrics...\n")
    sentiment_mapper = config.SENTIMENT_MAPPING
    
    # Use the enhanced evaluation
    macro_f1, weighted_f1 = evaluate_imbalanced_model(
        trained_model, test_loader, config.DEVICE
    )
    
    print(f"🎯 Final Test Scores:")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    
    evaluate_and_plot(
        trained_model,
        test_data,
        torch.nn.CrossEntropyLoss(),
        config.DEVICE,
        class_names=list(sentiment_mapper.values()),
        run_folder=config.MODEL_EVALUATION_OUTPUT_DIR,
    )


if __name__ == "__main__":
    # Entry point of the script
    main()
