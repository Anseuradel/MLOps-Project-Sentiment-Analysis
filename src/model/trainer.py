import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")

from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Dict, List
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight


from src.model.evaluate import evaluate
from src.model.model import SentimentClassifier
from config import MODEL_TRAINING_OUTPUT_DIR

from huggingface_hub import HfApi, HfFolder, upload_file

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight



def train_epoch(
    model: SentimentClassifier,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    # Loops through training DataLoader (batches of data)
    for batch in tqdm(data_loader, desc="Training"):
        # Sends data to GPU/CPU (input_ids, attention_mask, labels).
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        # Runs the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Computes loss (how wrong the predictions are)
        loss = loss_fn(outputs, labels)

        # Computes gradient
        loss.backward()

        # Updates model weigths
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(data_loader), correct_predictions / total_samples


# def train_model(
#     model: SentimentClassifier,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     device: torch.device,
#     epochs: int = 3,
#     lr: float = 2e-5,
#     run_folder: str = MODEL_TRAINING_OUTPUT_DIR,
# ):
#     model = model.to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#     scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=len(train_loader) * epochs,
#     )

def train_model(
    model: SentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
    run_folder: str = MODEL_TRAINING_OUTPUT_DIR,
):
    model = model.to(device)

    #  Compute class weights based on training dataset
    labels = train_loader.dataset.labels if hasattr(train_loader.dataset, "labels") else None
    if labels is not None:
        
        # Convert labels to a flat numpy array
        labels = np.array(train_loader.dataset.labels)
        unique_classes = np.unique(labels)
        
        # 🧮 Compute class weights safely
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels
        )
        
        # 🛠 Ensure all expected classes (e.g. 0–5) are present
        num_classes = model.fc.out_features if hasattr(model, "fc") else len(unique_classes)
        if len(unique_classes) < num_classes:
            print(f"⚠️ Missing some classes in current chunk — filling weights to {num_classes}.")
            full_weights = np.ones(num_classes)
            for i, cls in enumerate(unique_classes):
                full_weights[int(cls)] = class_weights[i]
            class_weights = full_weights
        
        # Convert to tensor
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"🧮 Using class weights: {class_weights}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                                
        
        print(f" Using class weights: {class_weights}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print(" Could not find dataset labels, using unweighted CrossEntropyLoss.")
        loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs,
    )

    # Create timestamped folder for this training run
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(run_folder, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    best_val_acc = 0
    best_model_path = os.path.join("outputs", "best_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'-' * 10}")

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Accuracy: {val_acc:.4f}\n")

        # Save metrics for plotting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save locally if it's the best so far
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved locally: {best_model_path}\n")
            best_val_acc = val_acc

    # --- Push once, at the end of training ---
    try:
        repo_id = "Adelanseur/MLOps-Project"
        upload_file(
            path_or_fileobj=best_model_path,
            path_in_repo="best_model.pth",  # overwrite same file
            repo_id=repo_id,
            token=HfFolder.get_token()
        )
        print(f"✅ Final best model uploaded to Hugging Face Hub: {repo_id}/best_model.pth")
    except Exception as e:
        print(f"⚠️ Failed to push model to Hugging Face Hub: {e}")

    # Save training history as JSON
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📄 Saved Training History: {history_path}\n")

    plot_training_results(history, run_dir)

    return model


def plot_training_results(history: Dict[str, List[float]], run_dir: str):
    """
    Plots the training & validation accuracy and loss curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # ✅ Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()

    # ✅ Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid()

    # ✅ Save plots
    accuracy_and_loss_plot_path = os.path.join(run_dir, "accuracy_and_loss_plot.png")

    plt.savefig(accuracy_and_loss_plot_path)

    print(f"📊 Saved Accuracy and Loss Plot: {accuracy_and_loss_plot_path}\n")
