import pandas as pd
import json
from config import SENTIMENT_MAPPING, LABEL_MAPPING

def load_file_by_type(file_path):
    """
    Loads file depending on its extension (csv, xlsx, json, jsonl, txt).
    Supports JSON lines format used by Amazon review datasets.
    
    Args:
        file_path (str): Path to the file to be loaded
        
    Returns:
        pandas.DataFrame: Loaded data as a DataFrame
        
    Raises:
        ValueError: If file format is unsupported
        FileNotFoundError: If file doesn't exist
    """
    try:
        # Handle CSV and TXT files (assuming CSV format)
        if file_path.endswith(".csv") or file_path.endswith(".txt"):
            return pd.read_csv(file_path)
        
        # Handle JSON Lines format (common in Amazon datasets)
        elif file_path.endswith(".jsonl"):
            # JSON Lines: each line is a separate JSON object
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Parse each line as individual JSON object
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid JSON lines but continue processing
                        continue
            return pd.DataFrame(data)
        
        # Handle standard JSON files
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        # Handle Excel files
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path, engine="openpyxl")
        
        # Raise error for unsupported file formats
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {file_path} not found.")


def map_rating_to_label(rating):
    """
    Maps 1–5 star ratings to sentiment categories and numeric labels.
    Adjust to match config.N_CLASSES.
    
    Args:
        rating (int/float): Rating value (1-5 stars)
        
    Returns:
        tuple: (sentiment_text, numeric_label) 
               e.g., ("positive", 3) for 4-star rating
    """
    # Convert to integer (rounding if necessary)
    rating = int(round(rating))
    
    # Define mapping from ratings to sentiment labels and numeric IDs
    mapping = {
        1: ("very negative", 0),  # 1 star = very negative
        2: ("negative", 1),       # 2 stars = negative  
        3: ("neutral", 2),        # 3 stars = neutral
        4: ("positive", 3),       # 4 stars = positive
        5: ("very positive", 4),  # 5 stars = very positive
    }
    
    # Return mapped value or default to neutral if rating is out of range
    return mapping.get(rating, ("neutral", 2))


def load_data(file_path, merge_labels=True):
    """
    Loads and processes Amazon Gift Card review data.
    Expects columns like: rating, title, text.
    
    Args:
        file_path (str): Path to the data file
        merge_labels (bool): Whether to merge title and text (legacy parameter)
        
    Returns:
        pandas.DataFrame: Processed data with columns: text, label_id, label_text
        
    Raises:
        ValueError: If dataset doesn't have required columns or has invalid labels
    """
    # Load raw data from file
    df = load_file_by_type(file_path)

    # ---- Handle dataset with rating and text columns (Amazon format) ----
    if {"rating", "text"}.issubset(df.columns):
        # Merge title + text if title exists and is not empty
        df["text"] = df.apply(
            lambda x: f"{x.get('title', '')}. {x['text']}" if "title" in x and pd.notna(x["title"]) else x["text"],
            axis=1
        )

        # Remove rows with missing text or rating
        df = df.dropna(subset=["text", "rating"])
        
        # Apply rating to label mapping
        df["label_text"], df["label_id"] = zip(*df["rating"].apply(map_rating_to_label))
        
        # Select and reorder final columns
        df = df[["text", "label_id", "label_text"]]

    # ---- Handle legacy CSV with 'text' and 'label' columns ----
    elif {"text", "label"}.issubset(df.columns):
        # Keep only text and label columns, remove missing values
        df = df[["text", "label"]].dropna()
        
        # Validate that all labels are in the allowed set
        if not df["label"].isin(LABEL_MAPPING.keys()).all():
            raise ValueError(
                f"Dataset contains invalid label values. Allowed values: {sorted(LABEL_MAPPING.keys())}"
            )
        
        # Map string labels to numeric IDs and sentiment text
        df["label_id"] = df["label"].map(LABEL_MAPPING).astype(int)
        df["label_text"] = df["label_id"].map(SENTIMENT_MAPPING)
        
        # Select final columns
        df = df[["text", "label_id", "label_text"]]

    else:
        # Raise error if required columns are missing
        raise ValueError("Dataset must contain either ('rating', 'text') or ('text', 'label') columns.")

    return df