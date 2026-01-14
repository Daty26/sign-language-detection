import csv
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../sign_language_detection
sys.path.append(str(PROJECT_ROOT))

from features.feature_extractor import extract_features


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_dataset_images(dataset_dir: Path):
    """
    Expects:
      dataset_dir/
        A/*.png
        B/*.png
        ...
    Yields: (label, image_path)
    """
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


def main():
    dataset_dir = PROJECT_ROOT / "dataset"
    out_csv = PROJECT_ROOT / "dataset" / "dataset.csv"

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,  # lower helps small/low-quality images
        min_tracking_confidence=0.5
    )

    num_features = 20  # your extractor output size
    write_header_if_needed(out_csv, num_features)

    total = 0
    saved = 0
    skipped_no_hand = 0
    skipped_bad_img = 0

    with out_csv.open("a", newline="") as f:
        writer = csv.writer(f)

        for label, img_path in iter_dataset_images(dataset_dir):
            total += 1

            img = cv2.imread(str(img_path))
            if img is None:
                skipped_bad_img += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                skipped_no_hand += 1
                continue

            hand_lm = result.multi_hand_landmarks[0].landmark
            hand_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

            feat = extract_features(hand_np)
            if feat is None or len(feat) != num_features:
                skipped_no_hand += 1
                continue

            writer.writerow([label] + list(map(float, feat)))
            saved += 1

            # Progress every 200
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
