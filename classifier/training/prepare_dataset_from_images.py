# classifier/training/prepare_dataset_from_images.py
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from features.feature_extractor import extract_features

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_dataset_images(dataset_dir: Path):
    for label_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield label, p


def write_header_if_needed(csv_path: Path, num_features: int):
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + [f"f{i}" for i in range(num_features)])


def _create_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def extract_features_from_image_path(
    hands,
    img_path: Path,
    num_features: int = 20,
) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand_lm = result.multi_hand_landmarks[0].landmark
    hand_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

    feat = extract_features(hand_np)
    if feat is None or len(feat) != num_features:
        return None

    return np.asarray(feat, dtype=np.float32)


def build_features_for_paths(
    rows: Iterable[Tuple[str, Path]],
    num_features: int = 20,
    cache_csv: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    rows: iterable of (label, img_path)

    Returns:
      X: [N, num_features]
      y: [N] labels (strings)
      stats: counters
    """
    stats = {"total": 0, "saved": 0, "skipped_no_hand": 0, "skipped_bad_img": 0}

    # Optional cache (simple): if cache_csv exists, load and return it
    if cache_csv is not None and cache_csv.exists():
        import pandas as pd
        df = pd.read_csv(cache_csv)
        y = df["label"].astype(str).to_numpy()
        X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
        stats["total"] = len(df)
        stats["saved"] = len(df)
        return X, y, stats

    hands = _create_hands()
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    try:
        for label, img_path in rows:
            stats["total"] += 1

            feat = extract_features_from_image_path(hands, img_path, num_features=num_features)
            if feat is None:
                stats["skipped_no_hand"] += 1
                continue

            X_list.append(feat)
            y_list.append(label)
            stats["saved"] += 1

            if stats["total"] % 200 == 0:
                print(f"Processed {stats['total']} | saved {stats['saved']} | no_hand {stats['skipped_no_hand']}")
    finally:
        hands.close()

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, num_features), dtype=np.float32)
    y = np.array(y_list, dtype=object)

    # Save cache if requested
    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        with cache_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label"] + [f"f{i}" for i in range(num_features)])
            for lbl, feat in zip(y_list, X_list):
                w.writerow([lbl] + list(map(float, feat)))

    return X, y, stats


def main():
    dataset_dir = PROJECT_ROOT / "dataset"
    out_csv = PROJECT_ROOT / "dataset" / "dataset.csv"

    num_features = 20
    write_header_if_needed(out_csv, num_features)

    total = 0
    saved = 0
    skipped_no_hand = 0
    skipped_bad_img = 0

    hands = _create_hands()

    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)

        for label, img_path in iter_dataset_images(dataset_dir):
            total += 1
            feat = extract_features_from_image_path(hands, img_path, num_features=num_features)

            if feat is None:
                skipped_no_hand += 1
                continue

            writer.writerow([label] + list(map(float, feat)))
            saved += 1

            if total % 200 == 0:
                print(f"Processed {total} | saved {saved} | no_hand {skipped_no_hand} | bad_img {skipped_bad_img}")

    hands.close()

    print("\nDone.")
    print(f"Total images: {total}")
    print(f"Saved rows:   {saved}")
    print(f"Skipped no hand: {skipped_no_hand}")
    print(f"Skipped bad img: {skipped_bad_img}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
