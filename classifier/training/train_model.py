import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from classifier.ml_classifier import MLClassifier
from prepare_dataset import load_dataset


def train_knn(X, y, k=7, model_path="../model.pkl"):
    print(f"\n Training KNN classifier with k={k}...")
    clf = MLClassifier(path=model_path)
    clf.train(X, y, k=k)
    print("KNN training complete.")
    return clf


def train_random_forest(X, y, model_path="../model.pkl"):
    print("\nTraining RandomForest classifier...")
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    model = RandomForestClassifier(n_estimators=150)

    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("RandomForest training complete.")
    return model


def main():
    print("\nLoading dataset...")
    X, y = load_dataset()
    X = np.array(X)
    y = np.array(y)

    print(f" Dataset loaded: {len(X)} samples, {len(set(y))} classes")

    # Shuffle + split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    print(f" Split: {len(X_train)} train / {len(X_test)} test")

    #choose model here:
    USE_RANDOM_FOREST = False  # make True if you want rf instead of knn

    model_path = "../model.pkl"

    # Train model
    if USE_RANDOM_FOREST:
        model = train_random_forest(X_train, y_train, model_path)
        predictions = model.predict(X_test)
    else:
        clf = MLClassifier(path=model_path)
        clf.train(X_train, y_train, k=7)
        predictions = [clf.predict(vec) for vec in X_test]

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n Accuracy: {accuracy * 100:.2f}%")
    print(f" Model saved to: {model_path}")


if __name__ == "__main__":
    main()
