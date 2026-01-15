# classifier/evaluation/make_split.py
import os
import pandas as pd
from datasets import load_dataset, Image

DATA_DIR = "dataset"
OUT_CSV = "splits/split_seed42.csv"
SEED = 42
TEST_SIZE = 0.2

def main():
    ds = load_dataset("imagefolder", data_dir=DATA_DIR)["train"]

    # IMPORTANT: keep file paths instead of decoding into PIL
    ds = ds.cast_column("image", Image(decode=False))

    split = ds.train_test_split(test_size=TEST_SIZE, seed=SEED, stratify_by_column="label")
    train_ds, test_ds = split["train"], split["test"]

    label_feature = ds.features["label"]  # ClassLabel

    def to_df(d, split_name):
        return pd.DataFrame({
            "path": [ex["image"]["path"] for ex in d],
            "label": [label_feature.int2str(int(ex["label"])) for ex in d],
            "split": split_name,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.concat([to_df(train_ds, "train"), to_df(test_ds, "test")], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV, "rows:", len(df))

if __name__ == "__main__":
    main()
