import os
import pandas as pd
from datasets import load_dataset, Image
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "dataset"
OUT_CSV = "splits/folds_seed42_k5.csv"
SEED = 42
K = 5

def main():
    ds = load_dataset("imagefolder", data_dir=DATA_DIR)["train"]
    ds = ds.cast_column("image", Image(decode=False))

    label_feature = ds.features["label"]  # ClassLabel

    paths = [ex["image"]["path"] for ex in ds]
    y_int = [int(ex["label"]) for ex in ds]  # ints for stratified k-fold
    y_str = [label_feature.int2str(i) for i in y_int]

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    fold_col = [-1] * len(paths)
    for fold_idx, (_, test_idx) in enumerate(skf.split(paths, y_int)):
        for i in test_idx:
            fold_col[i] = fold_idx

    df = pd.DataFrame({
        "path": paths,
        "label": y_str,
        "fold": fold_col,
    })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} rows: {len(df)} | folds: {sorted(df['fold'].unique())}")

if __name__ == "__main__":
    main()
