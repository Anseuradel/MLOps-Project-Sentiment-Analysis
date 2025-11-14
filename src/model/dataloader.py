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
        review = str(self.reviews[idx])  
        label = int(self.labels[idx])     

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)  
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
    #Convert labels to tensor
    labels = torch.tensor(df["label_id"].astype(int).to_numpy(), dtype=torch.long)

    dataset = SentimentDataset(
        reviews=df["text"].to_numpy(),
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    ) 

## this function is a dataloader that implements weighted sampler 

# def create_dataloader(df, tokenizer, max_len, batch_size, use_weighted_sampler=True):
#     """
#     Create a DataLoader for the given dataframe.
#     Optionally applies WeightedRandomSampler to handle class imbalance.
#     """
#     # Convert labels to tensor
#     labels = torch.tensor(df["label_id"].astype(int).to_numpy(), dtype=torch.long)

#     dataset = SentimentDataset(
#         reviews=df["text"].to_numpy(),
#         labels=labels,
#         tokenizer=tokenizer,
#         max_len=max_len,
#     )

#     if use_weighted_sampler:
#         # Compute class frequencies
#         class_counts = np.bincount(labels.numpy())
#         class_weights = 1.0 / class_counts

#         # Assign a weight to each sample
#         sample_weights = class_weights[labels.numpy()]

#         # Create the sampler
#         sampler = WeightedRandomSampler(
#             weights=torch.DoubleTensor(sample_weights),
#             num_samples=len(sample_weights),
#             replacement=True
#         )

#         # Use sampler instead of shuffle
#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#         )
#     else:
#         # Default: simple shuffled DataLoader
#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=True
#         )

#     return dataloader
