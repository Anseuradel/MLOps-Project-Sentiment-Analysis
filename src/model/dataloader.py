import os
import torch
import pandas as pd
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from transformers import PreTrainedTokenizerBase

class SentimentDataset(Dataset):
    """
    A custom PyTorch Dataset for handling sentiment analysis data.
    This class handles tokenization and formatting of text data for transformer models.
    """
    
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        """
        Initialize the SentimentDataset.
        
        Args:
            reviews (array-like): List or array of text reviews
            labels (array-like): List or array of corresponding labels
            tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer for text processing
            max_len (int): Maximum sequence length for tokenization
        """
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_ids: Token IDs from the tokenizer
                - attention_mask: Attention mask for the tokens
                - labels: Target label for the sample
        """
        # Ensure review is string type (handle potential NaN or non-string values)
        review = str(self.reviews[idx])  
        # Convert label to integer
        label = int(self.labels[idx])     

        # Tokenize the review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,    # Add [CLS] and [SEP] tokens
            max_length=self.max_len,    # Truncate/pad to max length
            return_token_type_ids=False, # Not needed for single-sequence tasks
            padding="max_length",       # Pad all sequences to max_length
            truncation=True,            # Truncate sequences longer than max_length
            return_attention_mask=True, # Generate attention mask
            return_tensors="pt"         # Return PyTorch tensors
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),        # Remove extra dimension: [1, seq_len] -> [seq_len]
            "attention_mask": encoding["attention_mask"].flatten(),  # Same flattening for attention mask
            "labels": torch.tensor(label, dtype=torch.long)      # Convert label to tensor with long dtype
        }


def create_dataloader(df, tokenizer, max_len, batch_size):
    """
    Create a DataLoader for training or evaluation.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'text' and 'label_id' columns
        tokenizer (PreTrainedTokenizerBase): Tokenizer for processing text
        max_len (int): Maximum sequence length for tokenization
        batch_size (int): Number of samples per batch
        
    Returns:
        DataLoader: PyTorch DataLoader ready for training/inference
    """
    # Convert labels to PyTorch tensor
    # Ensure labels are integers and convert to appropriate tensor format
    labels = torch.tensor(df["label_id"].astype(int).to_numpy(), dtype=torch.long)

    # Create the custom dataset
    dataset = SentimentDataset(
        reviews=df["text"].to_numpy(),  # Convert text column to numpy array
        labels=labels,                  # Pre-converted labels tensor
        tokenizer=tokenizer,            # Tokenizer for text processing
        max_len=max_len,                # Maximum sequence length
    )

    # Create and return the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True            # Shuffle data for training (set to False for validation/test)
    )