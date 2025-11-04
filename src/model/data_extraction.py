# import pandas as pd
# import json 

# from config import SENTIMENT_MAPPING, LABEL_MAPPING

# def load_file_by_type(file_path):

#     try:
#         if file_path.endswith(".csv") or file_path.endswith(".txt"):
#             return pd.read_csv(file_path)  # Load csv file
#         elif file_path.endswith(".json"):
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#             return pd.DataFrame(data)  # Convert json to dataframe
#         elif file_path.endswith(".xlsx"):
#             return pd.read_excel(file_path, engine="openpyxl")  # Load Excel file
#         else:
#             raise ValueError(
#                 f"Unsupported file format : {file_path}. Only CSV, TXT, JSON, and XLSX are supported."
#             )
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Error: File {file_path} not found.")


# def load_data(file_path, merge_labels):

#     try:
#         df = load_file_by_type(file_path)

#         # Check if required columns exist
#         required_columns = {"text", "label"}
#         if not required_columns.issubset(df.columns):
#             raise ValueError("Dataset must contain 'text' and 'label' columns.")

#         # Keep only relevant columns
#         df = df[["text", "label"]].dropna()

#         # ✅ Convert `score` values using `LABEL_MAPPING` (1-5 → 0-4)
#         if not df["label"].isin(LABEL_MAPPING.keys()).all():
#             raise ValueError(
#                 f"Dataset contains invalid score values. Allowed values: {sorted(LABEL_MAPPING.keys())}"
#             )

#         df["label_id"] = df["label"].map(LABEL_MAPPING).astype(int)
#         df["label_text"] = df["label_id"].map(SENTIMENT_MAPPING)

        
#         # Then only return the columns needed for training
#         return df[["text", "label_id", "label_text"]]
#         # return df

#     except FileNotFoundError as e:
#         raise FileNotFoundError(e)
#     except pd.errors.EmptyDataError:
#         raise ValueError(f"Error: File {file_path} is empty.")
#     except Exception as e:
#         raise ValueError(f"Unexpected error: {e}")


import pandas as pd
import json
from config import SENTIMENT_MAPPING, LABEL_MAPPING

def load_file_by_type(file_path):
    """
    Loads file depending on its extension (csv, xlsx, json, jsonl, txt).
    Supports JSON lines format used by Amazon review datasets.
    """
    try:
        if file_path.endswith(".csv") or file_path.endswith(".txt"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".jsonl"):
            # JSON Lines: each line is a separate JSON object
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return pd.DataFrame(data)
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {file_path} not found.")


def map_rating_to_label(rating):
    """
    Maps 1–5 star ratings to sentiment categories and numeric labels.
    Adjust to match config.N_CLASSES.
    """
    rating = int(round(rating))
    mapping = {
        1: ("very negative", 0),
        2: ("negative", 1),
        3: ("neutral", 2),
        4: ("positive", 3),
        5: ("very positive", 4),
    }
    return mapping.get(rating, ("neutral", 2))


def load_data(file_path, merge_labels=True):
    """
    Loads and processes Amazon Gift Card review data.
    Expects columns like: rating, title, text.
    """
    df = load_file_by_type(file_path)

    # ---- Handle dataset ----
    if {"rating", "text"}.issubset(df.columns):
        # Merge title + text
        df["text"] = df.apply(
            lambda x: f"{x.get('title', '')}. {x['text']}" if "title" in x and pd.notna(x["title"]) else x["text"],
            axis=1
        )

        # Convert ratings to sentiment labels
        df = df.dropna(subset=["text", "rating"])
        df["label_text"], df["label_id"] = zip(*df["rating"].apply(map_rating_to_label))
        df = df[["text", "label_id", "label_text"]]

    # ---- Handle legacy CSV with 'text' and 'label' ----
    elif {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].dropna()
        if not df["label"].isin(LABEL_MAPPING.keys()).all():
            raise ValueError(
                f"Dataset contains invalid label values. Allowed values: {sorted(LABEL_MAPPING.keys())}"
            )
        df["label_id"] = df["label"].map(LABEL_MAPPING).astype(int)
        df["label_text"] = df["label_id"].map(SENTIMENT_MAPPING)
        df = df[["text", "label_id", "label_text"]]

    else:
        raise ValueError("Dataset must contain either ('rating', 'text') or ('text', 'label') columns.")

    return df
