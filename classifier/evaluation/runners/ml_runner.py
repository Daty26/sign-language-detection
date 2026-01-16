# classifier/evaluation/runners/ml_runner.py
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from classifier.evaluation.load_split import load_split_csv
from classifier.training.train_model import train_model
from classifier.training.prepare_dataset_from_images import build_features_for_paths


def run(split_csv: Path, fold: Optional[int]) -> Dict[str, Any]:
    train_df, test_df, _class_names = load_split_csv(split_csv, fold=fold)
    # train_df, test_df, _class_names = load_split_csv(split_csv)


    # Build rows: (label, path)
    train_rows = [(row["label"], Path(row["path"])) for _, row in train_df.iterrows()]
    test_rows  = [(row["label"], Path(row["path"])) for _, row in test_df.iterrows()]

    # Cache per fold so it’s not re-extracted every run
    fold_tag = f"fold{fold}" if fold is not None else "split"
    cache_dir = Path("runs/cache/mediapipe_features")
    cache_train = cache_dir / f"train_{fold_tag}.csv"
    cache_test  = cache_dir / f"test_{fold_tag}.csv"

    X_train, y_train, stats_tr = build_features_for_paths(train_rows, cache_csv=cache_train)
    X_test, y_test, stats_te = build_features_for_paths(test_rows, cache_csv=cache_test)

    # Important: Some images may be skipped (no hand detected). This is OK but we should report it.
    print(f"[ML] train kept {len(X_train)}/{stats_tr['total']} | test kept {len(X_test)}/{stats_te['total']}")

    # Save model somewhere safe (avoid overwriting production model.pkl)
    model_out = Path(f"runs/ml/{fold_tag}/model.pkl")
    model_out.parent.mkdir(parents=True, exist_ok=True)

    out = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_path=str(model_out),
        use_random_forest=False,
        k=7,
        return_details=False,  # keep runner output small
    )

    out.update({
        "model": "ml",
        "fold": fold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_skipped": int(stats_tr["total"] - stats_tr["saved"]),
        "test_skipped": int(stats_te["total"] - stats_te["saved"]),
    })
    return out
