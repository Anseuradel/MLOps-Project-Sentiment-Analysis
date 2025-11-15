import pandas as pd
import nlpaug.augmenter.word as naw
import nltk

# Download required NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

# Map label_text → label_id (consistent!)
LABEL_MAP = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4
}

def balance_dataset_with_augmentation(df, text_col="text", label_col="label_text"):

    aug = naw.SynonymAug(aug_src="wordnet")

    # Ensure label_id exists
    if "label_id" not in df.columns:
        df["label_id"] = df[label_col].map(LABEL_MAP)

    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    print("Class counts before augmentation:")
    print(class_counts)
    print()

    augmented_rows = []

    for label, count in class_counts.items():

        if count < max_count:

            df_class = df[df[label_col] == label]
            needed = max_count - count

            print(f"Augmenting class '{label}' with {needed} new samples...")

            augmented_texts = []
            while len(augmented_texts) < needed:
                sample = df_class.sample(1, replace=True).iloc[0]
                augmented_text = aug.augment(sample[text_col])
                if isinstance(augmented_text, list):
                    augmented_text = augmented_text[0]
                augmented_texts.append(augmented_text)

            # Create augmented samples
            new_rows = pd.DataFrame({
                text_col: augmented_texts[:needed],
                label_col: [label] * needed,
                "label_id": [LABEL_MAP[label]] * needed   # <-- FIXED
            })

            augmented_rows.append(new_rows)

    if augmented_rows:
        df_augmented = pd.concat(augmented_rows, ignore_index=True)
        df_balanced = pd.concat([df, df_augmented], ignore_index=True)
    else:
        df_balanced = df.copy()

    print("\n✅ Class counts after augmentation:")
    print(df_balanced[label_col].value_counts())
    print()

    # Remove rows missing text or label_text only
    df_balanced = df_balanced.dropna(subset=[text_col, label_col, "label_id"])
    df_balanced = df_balanced.reset_index(drop=True)

    # Ensure label_id is int with no NaNs
    df_balanced["label_id"] = df_balanced["label_id"].astype(int)

    return df_balanced
