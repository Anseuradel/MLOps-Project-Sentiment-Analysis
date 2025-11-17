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


def train_model(
    model: SentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
    run_folder: str = MODEL_TRAINING_OUTPUT_DIR,
) -> SentimentClassifier:
    """
    Main training function that handles the complete training loop.
    Includes class weighting, learning rate scheduling, model checkpointing,
    and Hugging Face Hub integration.
    
    Args:
        model: The sentiment classifier model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (CPU/GPU)
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        run_folder: Directory to save training outputs
        
    Returns:
        SentimentClassifier: The trained model
    """
    # Ensure model is on the correct device
    model = model.to(device)

    # Compute class weights based on training dataset to handle class imbalance
    labels = train_loader.dataset.labels if hasattr(train_loader.dataset, "labels") else None
    if labels is not None:
        
        # Convert labels to a flat numpy array for weight computation
        labels = np.array(train_loader.dataset.labels)
        unique_classes = np.unique(labels)
        
        # Compute class weights using sklearn's balanced method
        # This gives higher weight to underrepresented classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels
        )
        
        # Handle case where current chunk doesn't contain all expected classes
        num_classes = model.fc.out_features if hasattr(model, "fc") else len(unique_classes)
        if len(unique_classes) < num_classes:
            print(f"⚠️ Missing some classes in current chunk — filling weights to {num_classes}.")
            full_weights = np.ones(num_classes)
            for i, cls in enumerate(unique_classes):
                full_weights[int(cls)] = class_weights[i]
            class_weights = full_weights
        
        # Convert numpy weights to PyTorch tensor and move to device
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"🧮 Using class weights: {class_weights}")
        
        # Initialize loss function with class weights for imbalanced data
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Fallback to unweighted loss if labels aren't available
        print(" Could not find dataset labels, using unweighted CrossEntropyLoss.")
        loss_fn = nn.CrossEntropyLoss()

    # Initialize optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # Initialize learning rate scheduler with linear decay
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs,
    )

    # Create timestamped folder for this training run for organization
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_dir = os.path.join(run_folder, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Track best validation accuracy for model checkpointing
    best_val_acc = 0
    best_model_path = os.path.join("outputs", "best_model.pth")
    
    # Initialize history dictionary to track training progress
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Main training loop over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'-' * 10}")

        # Train for one epoch and get metrics
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, device)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Accuracy: {val_acc:.4f}\n")

        # Save metrics for plotting and analysis
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save model checkpoint if it achieves best validation accuracy
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved locally: {best_model_path}\n")
            best_val_acc = val_acc

    # --- Push final best model to Hugging Face Hub ---
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

    # Save training history as JSON for later analysis
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📄 Saved Training History: {history_path}\n")

    # Generate and save training plots
    plot_training_results(history, run_dir)

    return model


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
    
    print("🎯 Class weights computed:", class_weights)
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
    
    print(f"📊 Imbalance-Aware Evaluation:")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']))
    
    return macro_f1, weighted_f1

def train_model_with_imbalance_handling(
    model: SentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    df_train: pd.DataFrame,
    epochs: int = 3,
    lr: float = 2e-5,
    use_focal_loss: bool = True,
    use_class_weights: bool = True
):
    """Enhanced training with imbalance handling."""
    
    # 1. Compute class weights from training data
    if use_class_weights:
        class_weights = get_class_weights(df_train, label_column='label_id')
        class_weights = class_weights.to(device)
        print(f"🎯 Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # 2. Choose loss function
    if use_focal_loss and use_class_weights:
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        print("🎯 Using Focal Loss with class weights")
    elif use_class_weights:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("🎯 Using Weighted CrossEntropy Loss")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        print("🎯 Using Standard CrossEntropy Loss")
    
    # 3. Setup optimizer and scheduler (same as before)
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

    best_val_f1 = 0  # Track F1 instead of accuracy
    best_model_path = os.path.join("outputs", "best_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_macro_f1": [], "val_weighted_f1": []}

    # 4. Training loop with imbalance-aware validation
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'-' * 10}")

        # Train epoch (same as before)
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        
        # 🎯 NEW: Validate with F1 scores instead of accuracy
        val_macro_f1, val_weighted_f1 = evaluate_imbalanced_model(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f}, Val Weighted F1: {val_weighted_f1:.4f}\n")

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_weighted_f1"].append(val_weighted_f1)

        # Save best model based on macro F1
        if val_macro_f1 > best_val_f1:
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved (F1: {val_macro_f1:.4f}): {best_model_path}\n")
            best_val_f1 = val_macro_f1

    # Save training history
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📄 Saved Training History: {history_path}\n")

    # Plot training results
    plot_training_results(history, run_dir)

    return model

def plot_training_results(history: Dict[str, List[float]], run_dir: str):
    """
    Plots the training & validation accuracy and loss curves.
    Saves the plots to the specified directory.
    
    Args:
        history: Dictionary containing training history metrics
        run_dir: Directory to save the generated plots
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # ✅ Plot 1: Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()

    # ✅ Plot 2: Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid()

    # ✅ Save plots to file
    accuracy_and_loss_plot_path = os.path.join(run_dir, "accuracy_and_loss_plot.png")
    plt.savefig(accuracy_and_loss_plot_path)

    print(f"📊 Saved Accuracy and Loss Plot: {accuracy_and_loss_plot_path}\n")
