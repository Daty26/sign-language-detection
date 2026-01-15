import pandas as pd

def load_split_csv(path="splits/split_seed42.csv"):
    df = pd.read_csv(path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    classes = sorted(df["label"].unique())
    return train_df, test_df, classes
