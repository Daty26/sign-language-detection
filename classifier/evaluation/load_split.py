import pandas as pd
from typing import Optional, Tuple, List

def load_split_csv(
    path="splits/split_seed42.csv",
    fold: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(path)

    classes = sorted(df["label"].unique())

    # Case 1: fold-based split (k-fold CV)
    if "fold" in df.columns:
        if fold is None:
            raise ValueError(
                "CSV contains 'fold' column, but no fold was specified. "
                "Pass fold=int to load_split_csv(...)."
            )

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        test_df  = df[df["fold"] == fold].reset_index(drop=True)
        return train_df, test_df, classes

    # Case 2: simple train/test split
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)
        return train_df, test_df, classes
    
    raise ValueError(
        f"{path} must contain either 'split' or 'fold' column."
    )
