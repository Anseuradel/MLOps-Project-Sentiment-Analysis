import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords

import config

import re 
import regex
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Initialize the tokenizer from Hugging Face transformers
# Uses the tokenizer name specified in config file (e.g., 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

def clean_text(text):
    """
    Cleans and preprocesses text data by performing several normalization steps.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove URLs (http, https, www links)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove punctuation and special characters
    # \W matches any non-word character (equivalent to [^a-zA-Z0-9_])
    text = re.sub(r"\W", " ", text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    # Replace with single space
    text = re.sub(r"\s+", " ", text)
    
    # Remove emojis using regex with unicode property support
    # \p{Emoji} matches any emoji character
    text = regex.compile(r'\p{Emoji}').sub('', text)
    
    # Optional: Remove stopwords (commented out for flexibility)
    # text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text


def tokenize_texts(texts, max_length):
    """
    Tokenizes a list of texts using the pre-trained tokenizer.
    Converts text to format suitable for transformer models.
    
    Args:
        texts (list): List of text strings to tokenize
        max_length (int): Maximum sequence length for truncation/padding
        
    Returns:
        dict: Dictionary containing tokenized outputs:
              - input_ids: Token IDs representing the text
              - attention_mask: Mask indicating which tokens to attend to
    """
    # Tokenize the texts with appropriate parameters for model input
    tokenized = tokenizer(
        texts, 
        padding=True,           # Pad sequences to same length
        truncation=True,        # Truncate sequences longer than max_length
        max_length=max_length,  # Maximum sequence length
        return_tensors="pt"     # Return PyTorch tensors
    )

    # Convert PyTorch tensors to lists for compatibility with Pandas DataFrames
    return {
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    }


def preprocess_data(df, test_size, max_length):
    """
    Main preprocessing pipeline that cleans, tokenizes, and splits the data.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with 'text' and 'label_id' columns
        test_size (float): Proportion of data to use for validation (0.0-1.0)
        max_length (int): Maximum sequence length for tokenization
        
    Returns:
        tuple: (train_df, val_df) - Training and validation DataFrames
    """
    # Clean the text data by applying the clean_text function to each row
    df["text"] = df["text"].apply(clean_text)

    # Tokenize the cleaned text data
    tokenized_data = tokenize_texts(df["text"].tolist(), max_length)

    # Add tokenized columns back to the DataFrame
    df["input_ids"] = tokenized_data["input_ids"]
    df["attention_mask"] = tokenized_data["attention_mask"]

    # Determine if stratified sampling is possible
    # Stratification requires at least 2 samples per class in both splits
    if df["label_id"].value_counts().min() < 2:
        # If any class has fewer than 2 samples, don't use stratification
        stratify_param = None
    else:
        # Use label distribution for stratified sampling
        stratify_param = df["label_id"]

    # Split data into training and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size,      # Proportion for validation set
        stratify=stratify_param,  # Maintain class distribution in splits
        random_state=42           # Seed for reproducible splits
    )

    return train_df, val_df