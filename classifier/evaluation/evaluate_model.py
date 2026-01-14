import os
import cv2
import numpy as np
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from classifier.evaluation.feature_pipeline import FeaturePipeline
from classifier.ml_classifier import MLClassifier

def load_dataset_paths(root_dir):
    samples = []
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(label_dir, fname), label))
    return samples

def main():
    dataset_root = "dataset"  # change if needed

    pipeline = FeaturePipeline()

    clf = MLClassifier(path="classifier/model.pkl")
    clf.load()

    y_true, y_pred = [], []
    skipped = 0

    samples = load_dataset_paths(dataset_root)
    print(f"Found {len(samples)} images")

    for path, label in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        
        img = pipeline.ensure_min_size(img, min_side=320)
        # img = pipeline.preprocess_for_detection(img)
        vec = pipeline.extract_from_bgr(img)
        if vec is None:
            skipped += 1
            continue

        pred = clf.predict(vec)
        y_true.append(label)
        y_pred.append(pred)

    print(f"Evaluated: {len(y_true)} | Skipped (no hand): {skipped}")
    if not y_true:
        print("No usable samples. Check detection / dataset images.")
        return

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nLabels:", labels)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
