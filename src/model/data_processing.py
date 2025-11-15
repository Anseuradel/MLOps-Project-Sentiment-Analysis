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
tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

def clean_text(text):
    """
    Cleans and preprocesses text data by performing several normalization steps.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned and normalized text
    """
    text = text.lower()                                # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text) # remove URLs
    text = re.sub(r"\W", " ", text)                     # remove punctuation
    text = re.sub(r"\s+", " ", text)                    # remove extra spaces
    text = regex.compile(r'\p{Emoji}').sub('', text)  # remove emoticones
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
  
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )

    # Convert PyTorch tensors to lists so they work with Pandas
    return {
      "input_ids": tokenized["input_ids"].tolist(),
      "attention_mask": tokenized["attention_mask"].tolist(),
    }

def preprocess_data(df, test_size, max_length, label_col="label_text"):
    """
    Preprocess pipeline: clean text, tokenize, create label_id, split.
    """

    # --- Safety check: ensure label_id exists ---
    if "label_id" not in df.columns:
        raise ValueError("❌ preprocess_data: 'label_id' column missing — augmentation must create it!")

    # --- Safety check: ensure no NaNs in label_id ---
    if df["label_id"].isna().any():
        print("⚠ Found NaN label_id, dropping them...")
        df = df.dropna(subset=["label_id"]).reset_index(drop=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Create numeric label_id BEFORE using stratify
    df["label_id"] = df[label_col].astype("category").cat.codes

    # Tokenize
    tokenized = tokenize_texts(df["text"].tolist(), max_length)
    df["input_ids"] = tokenized["input_ids"]
    df["attention_mask"] = tokenized["attention_mask"]

    # Determine if stratification is possible
    if df["label_id"].value_counts().min() < 2:
        stratify_param = None
    else:
        stratify_param = df["label_id"]

    # Split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_param,
        random_state=42
    )

    return train_df, val_df

