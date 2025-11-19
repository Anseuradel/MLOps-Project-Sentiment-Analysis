import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import matplotlib

# Set matplotlib backend to Agg for non-interactive plotting (useful for servers)
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
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


def train_epoch(
    model: SentimentClassifier,
    data_loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Trains the model for one complete epoch.
    
    Args:
        model: The sentiment classifier model to train
        data_loader: DataLoader providing training batches
        loss_fn: Loss function for computing training loss
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        device: Device to train on (CPU/GPU)
        
    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch
    """
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    # Loops through training DataLoader (batches of data)
    for batch in tqdm(data_loader, desc="Training"):
        # Sends data to GPU/CPU (input_ids, attention_mask, labels).
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Runs the model forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Computes loss (how wrong the predictions are)
        loss = loss_fn(outputs, labels)

        # Computes gradients via backpropagation
        loss.backward()

        # Updates model weights using computed gradients
        optimizer.step()
        
        # Updates learning rate according to scheduler
        scheduler.step()

        # Accumulate metrics for epoch statistics
        total_loss += loss.item()
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    # Return average loss and accuracy for the epoch
    return total_loss / len(data_loader), correct_predictions / total_samples

from sklearn.metrics import f1_score, classification_report

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_class_weights(df, label_column='label_id'):
    """Compute class weights for imbalanced dataset."""
    labels = df[label_column].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    print("Class weights computed:", class_weights)
    return torch.tensor(class_weights, dtype=torch.float)

def evaluate_imbalanced_model(model, test_loader, device):
    """Evaluate model with imbalanced-aware metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Use macro F1 instead of accuracy for imbalanced data
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Imbalance-Aware Evaluation:")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']))
    
    return macro_f1, weighted_f1

# def train_model_with_imbalance_handling(
#     model: SentimentClassifier,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     device: torch.device,
#     df_train: pd.DataFrame,
#     epochs: int = 3,
#     lr: float = 2e-5,
#     use_focal_loss: bool = True,
#     use_class_weights: bool = True
# ):
#     """Enhanced training with imbalance handling."""
    
#     # 1. Compute class weights from training data
#     if use_class_weights:
#         class_weights = get_class_weights(df_train, label_column='label_id')
#         class_weights = class_weights.to(device)
#         print(f"Using class weights: {class_weights}")
#     else:
#         class_weights = None
    
#     # 2. Choose loss function
#     if use_focal_loss and use_class_weights:
#         loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
#         print("Using Focal Loss with class weights")
#     elif use_class_weights:
#         loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
#         print("Using Weighted CrossEntropy Loss")
#     else:
#         loss_fn = torch.nn.CrossEntropyLoss()
#         print("Using Standard CrossEntropy Loss")
    
#     # 3. Setup optimizer and scheduler (same as before)
#     optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#     scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=len(train_loader) * epochs,
#     )

#     # Create timestamped folder for this training run
#     timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
#     run_dir = os.path.join(MODEL_TRAINING_OUTPUT_DIR, f"run_{timestamp}")
#     os.makedirs(run_dir, exist_ok=True)

#     best_val_f1 = 0  # Track F1 instead of accuracy
#     best_model_path = os.path.join("outputs", "best_model.pth")
#     history = {"train_loss": [], "train_acc": [], "val_macro_f1": [], "val_weighted_f1": []}

#     # 4. Training loop with imbalance-aware validation
#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
#         print(f"{'-' * 10}")

#         # Train epoch (same as before)
#         train_loss, train_acc = train_epoch(
#             model, train_loader, loss_fn, optimizer, scheduler, device
#         )
        
#         # NEW: Validate with F1 scores instead of accuracy
#         val_macro_f1, val_weighted_f1 = evaluate_imbalanced_model(model, val_loader, device)

#         print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
#         print(f"Val Macro F1: {val_macro_f1:.4f}, Val Weighted F1: {val_weighted_f1:.4f}\n")

#         # Save metrics
#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["val_macro_f1"].append(val_macro_f1)
#         history["val_weighted_f1"].append(val_weighted_f1)

#         # Save best model based on macro F1
#         if val_macro_f1 > best_val_f1:
#             torch.save(model.state_dict(), best_model_path)
#             print(f"New best model saved (F1: {val_macro_f1:.4f}): {best_model_path}\n")
#             best_val_f1 = val_macro_f1

#     # Save training history
#     history_path = os.path.join(run_dir, "training_history.json")
#     with open(history_path, "w") as f:
#         json.dump(history, f, indent=4)
#     print(f"Saved Training History: {history_path}\n")

#     # Plot training results
#     plot_training_results(history, run_dir)

#     return model

def train_model_with_imbalance_handling(
    model: SentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    df_train: pd.DataFrame,
    epochs: int = 3,
    lr: float = 2e-5,
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
    patience: int = 3  # NEW: Add patience parameter
):
    """Enhanced training with imbalance handling."""
    
    # 1. Compute class weights from training data
    if use_class_weights:
        class_weights = get_class_weights(df_train, label_column='label_id')
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # 2. Choose loss function
    if use_focal_loss and use_class_weights:
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        print("Using Focal Loss with class weights")
    elif use_class_weights:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropy Loss")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print("Using Standard CrossEntropy Loss")
    
    # 3. Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs,
    )

    # Create timestamped folder for this training run
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(MODEL_TRAINING_OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 🎯 NEW: Early stopping variables
    best_val_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join("outputs", "best_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_macro_f1": [], "val_weighted_f1": []}

    print(f"🎯 Training with early stopping (patience: {patience} epochs)")

    # 4. Training loop with early stopping
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'-' * 10}")

        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        
        # Validate with F1 scores
        val_macro_f1, val_weighted_f1 = evaluate_imbalanced_model(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f}, Val Weighted F1: {val_weighted_f1:.4f}")

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_weighted_f1"].append(val_weighted_f1)

        # 🎯 NEW: Early stopping logic
        if val_macro_f1 > best_val_f1:
            # Improvement found - save model and reset patience
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved (F1: {val_macro_f1:.4f}): {best_model_path}")
        else:
            # No improvement - increment patience counter
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            # Check if we should stop early
            if patience_counter >= patience:
                print(f"🏁 Early stopping triggered at epoch {epoch + 1}")
                print(f"🎯 Best validation F1: {best_val_f1:.4f}")
                break
        
        print()  # Empty line for readability

    # 🎯 NEW: Load the best model before returning
    print(f"🔄 Loading best model with F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save training history
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📄 Saved Training History: {history_path}")

    # Plot training results
    plot_training_results(history, run_dir)

    return model
    
def plot_training_results(history: Dict[str, List[float]], run_dir: str):
    """
    Plots the training & validation metrics curves.
    Updated for imbalance handling (F1 scores instead of loss/accuracy).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    
    # Check if val_loss exists (for backward compatibility)
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o", linewidth=2)
    elif "val_macro_f1" in history:
        # If we don't have val_loss, just plot train loss
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2, color='blue')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss", fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o", linewidth=2, color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy", fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 3: Validation F1 scores (NEW for imbalance handling)
    plt.subplot(2, 2, 3)
    if "val_macro_f1" in history:
        plt.plot(epochs, history["val_macro_f1"], label="Val Macro F1", marker="o", linewidth=2, color='red')
    if "val_weighted_f1" in history:
        plt.plot(epochs, history["val_weighted_f1"], label="Val Weighted F1", marker="o", linewidth=2, color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Scores", fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 4: Combined metrics overview
    plt.subplot(2, 2, 4)
    
    # Normalize metrics to same scale for comparison
    metrics_to_plot = []
    labels = []
    
    if "train_acc" in history:
        metrics_to_plot.append(history["train_acc"])
        labels.append("Train Accuracy")
    
    if "val_macro_f1" in history:
        metrics_to_plot.append(history["val_macro_f1"])
        labels.append("Val Macro F1")
    
    if "val_weighted_f1" in history:
        metrics_to_plot.append(history["val_weighted_f1"])
        labels.append("Val Weighted F1")
    
    for i, metric in enumerate(metrics_to_plot):
        plt.plot(epochs, metric, label=labels[i], marker="o", linewidth=2)
    
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Metrics Overview", fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(run_dir, "training_metrics_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved Training Metrics Plot: {plot_path}")
