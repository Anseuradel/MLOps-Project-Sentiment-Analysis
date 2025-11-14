import pandas as pd
import nlpaug.augmenter.word as naw
import nltk

# Download required NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

def balance_dataset_with_augmentation(df, text_col="text", label_col="label"):
    """
    Balance dataset by oversampling minority classes with NLP augmentation.
    Works with datasets containing columns: text, label
    """

    aug = naw.SynonymAug(aug_src="wordnet")

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
                aug_text = aug.augment(sample[text_col])
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                augmented_texts.append(aug_text)

            new_rows = pd.DataFrame({
                text_col: augmented_texts[:needed],
                label_col: [label] * needed
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

    # Clean any accidental NaN rows
    df_balanced = df_balanced.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    return df_balanced
