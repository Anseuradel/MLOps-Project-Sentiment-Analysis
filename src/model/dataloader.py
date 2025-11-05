import os
import torch
import pandas as pd
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from transformers import PreTrainedTokenizerBase

class SentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
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



# def create_dataloader(df, tokenizer, max_len, batch_size):
#     #Convert labels to tensor
#     labels = torch.tensor(df["label_id"].astype(int).to_numpy(), dtype=torch.long)

#     dataset = SentimentDataset(
#         reviews=df["text"].to_numpy(),
#         labels=labels,
#         tokenizer=tokenizer,
#         max_len=max_len,
#     )

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True
#     ) 

def create_dataloader(df, tokenizer, max_len, batch_size, use_weighted_sampler=True):
    """
    Create a DataLoader for the given dataframe.
    Optionally applies WeightedRandomSampler to handle class imbalance.
    """
    # Convert labels to tensor
    labels = torch.tensor(df["label_id"].astype(int).to_numpy(), dtype=torch.long)

    dataset = SentimentDataset(
        reviews=df["text"].to_numpy(),
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    if use_weighted_sampler:
        # Compute class frequencies
        class_counts = np.bincount(labels.numpy())
        class_weights = 1.0 / class_counts

        # Assign a weight to each sample
        sample_weights = class_weights[labels.numpy()]

        # Create the sampler
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        # Use sampler instead of shuffle
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
    else:
        # Default: simple shuffled DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    return dataloader
