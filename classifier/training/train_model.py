import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from classifier.ml_classifier import MLClassifier
from classifier.training.prepare_dataset import load_dataset
from classifier.training.config import DATASET_PATH, MODEL_PATH


def train_knn(X, y, k=7, model_path="../model.pkl"):
    print(f"\ntraining KNN classifier with k={k}")
    clf = MLClassifier(path=model_path, model_type="knn")
    clf.train(X, y, k=k)
    print("KNN training complete.")
    return clf


def train_random_forest(X, y, model_path="../model.pkl"):
    print("\ntraining RandomForest classifier")
    clf = MLClassifier(path=model_path, model_type="random_forest")
    clf.train(X, y, n_estimators=150)
    print("RandomForest training complete")
    return clf


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_path=MODEL_PATH,
    use_random_forest=False,
    k=7,
    n_estimators=150,
    return_details=True,
):
    """
    Train + evaluate using PROVIDED splits.
    This is what your benchmark runner will call.
    """
    if use_random_forest:
        clf = train_random_forest(X_train, y_train, n_estimators=n_estimators, model_path=model_path)
    else:
        clf = train_knn(X_train, y_train, k=k, model_path=model_path)

    predictions = [clf.predict(vec) for vec in X_test]
    acc = float(accuracy_score(y_test, predictions))

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nPer-class performance:")
    report = classification_report(y_test, predictions)

    print(report)
    print(f"\nModel saved to: {model_path}")

    if not return_details:
        return {"accuracy": acc}

    return {
        "accuracy": acc,
        "report": report,
        "y_true": y_test,
        "y_pred": predictions,
        "model_path": model_path,
        "use_random_forest": use_random_forest,
        "k": k,
        "n_estimators": n_estimators,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


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

    test_size = 0.2
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=seed
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    USE_RANDOM_FOREST = False
    model_path = MODEL_PATH

    train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_path=model_path,
        use_random_forest=USE_RANDOM_FOREST,
        k=7,
        n_estimators=150,
        return_details=True,
    )


if __name__ == "__main__":
    main()