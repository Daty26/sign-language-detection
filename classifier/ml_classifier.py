import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class MLClassifier:
    def __init__(self, path="classifier/model.pkl"):
        self.path = path
        self.model = None

    def train(self, X, y, k=5):
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X, y)
        self.save()

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features):
        arr = np.array(features).reshape(1, -1)
        return self.model.predict(arr)[0]
