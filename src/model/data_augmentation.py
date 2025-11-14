import pandas as pd
import nlpaug.augmenter.word as naw
import nltk

# Download required NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

def balance_dataset_with_augmentation(df, text_col="text", label_col="label_text"):
    """
    Balance dataset by oversampling minority classes with NLP augmentation.
    """

    aug = naw.SynonymAug(aug_src="wordnet")

    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    print("Class counts before augmentation:")
    print(class_counts)
    print()

    augmented_rows = []

    # Loop through each class
    for label, count in class_counts.items():
        if count < max_count:
            df_class = df[df[label_col] == label]
            needed = max_count - count
            print(f"Augmenting class '{label}' with {needed} new samples...")

            # Collect augmented examples
            augmented_texts = []
            while len(augmented_texts) < needed:
                # pick random samples from df_class
                sample = df_class.sample(1, replace=True).iloc[0]
                augmented_text = aug.augment(sample[text_col])
                if isinstance(augmented_text, list):
                    augmented_text = augmented_text[0]
                augmented_texts.append(augmented_text)

            # Create new rows for the augmented samples
            new_rows = pd.DataFrame({
                text_col: augmented_texts[:needed],
                label_col: [label] * needed
            })

            augmented_rows.append(new_rows)

    # Merge everything together
    if augmented_rows:
        df_augmented = pd.concat(augmented_rows, ignore_index=True)
        df_balanced = pd.concat([df, df_augmented], ignore_index=True)
    else:
        df_balanced = df.copy()

    print("\n✅ Class counts after augmentation:")
    print(df_balanced[label_col].value_counts())
    print()

    # ✅ Drop rows with missing labels (safety measure)
    df_balanced = df_balanced.dropna(subset=["label_id", "label_text"]).reset_index(drop=True)

    # ✅ Ensure label_id is integer
    df_balanced["label_id"] = df_balanced["label_id"].astype(int)

    df_augmented = df_augmented.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    return df_balanced
