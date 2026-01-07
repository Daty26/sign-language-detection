import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from classifier.ml_classifier import MLClassifier
from classifier.training.prepare_dataset import load_dataset
from classifier.training.config import DATASET_PATH, MODEL_PATH


def train_knn(X, y, k=7, model_path="../model.pkl"):
    print(f"\nTraining KNN classifier with k={k}...")
    clf = MLClassifier(path=model_path, model_type="knn")
    clf.train(X, y, k=k)
    print("KNN training complete.")
    return clf


def train_random_forest(X, y, model_path="../model.pkl"):
    print("\nTraining RandomForest classifier...")
    clf = MLClassifier(path=model_path, model_type="random_forest")
    clf.train(X, y, n_estimators=150)
    print("RandomForest training complete.")
    return clf


def main():
    print("\nLoading dataset...")
    X, y = load_dataset(DATASET_PATH)
    X = np.array(X)
    y = np.array(y)

    print(f"Dataset loaded: {len(X)} samples, {len(set(y))} classes")
    
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count} samples")

    # Shuffle + split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    USE_RANDOM_FOREST = False

    model_path = MODEL_PATH

    if USE_RANDOM_FOREST:
        clf = train_random_forest(X_train, y_train, model_path)
    else:
        clf = train_knn(X_train, y_train, k=7, model_path=model_path)
    
    predictions = [clf.predict(vec) for vec in X_test]
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nPer-class performance:")
    print(classification_report(y_test, predictions))
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()