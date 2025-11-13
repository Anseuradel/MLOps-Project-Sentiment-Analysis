import pandas as pd
import nlpaug.augmenter.word as naw


def balance_dataset_with_augmentation(df: pd.DataFrame, label_col: str = "label_text", text_col: str = "text"):
    """
    Augment minority classes using nlpaug to balance dataset.

    Args:
        df (pd.DataFrame): The original dataset.
        label_col (str): The column containing labels.
        text_col (str): The column containing text.

    Returns:
        pd.DataFrame: Balanced dataset (original + augmented samples).
    """
    aug = naw.SynonymAug(aug_src='wordnet')

    # Compute class distribution
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()
    print(f"Class counts before augmentation:\n{class_counts}\n")

    augmented_frames = []
    for label, count in class_counts.items():
        df_class = df[df[label_col] == label]
        if count < max_count:
            needed = max_count - count
            augmented_texts = []
            for text in df_class[text_col]:
                new_samples = aug.augment(text, n=3)
                augmented_texts.extend(new_samples)
                if len(augmented_texts) >= needed:
                    break

            df_aug = df_class.sample(n=needed, replace=True).copy()
            df_aug[text_col] = augmented_texts[:needed]
            augmented_frames.append(df_aug)

    # Combine everything
    df_augmented = pd.concat([df] + augmented_frames, ignore_index=True)
    print(f"✅ Dataset balanced — total samples: {len(df_augmented)}")
    print(df_augmented[label_col].value_counts())

    return df_augmented
