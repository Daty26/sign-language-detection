# classifier/ml_classifier.py
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MLClassifier:
    def __init__(self, path="classifier/model.pkl", model_type="knn"):
        self.path = path
        self.model = None
        self.scaler = StandardScaler()  # Add scaling for better accuracy
        self.model_type = model_type

    def train(self, X, y, k=7, n_estimators=150):
        # Scale features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == "knn":
            # Use distance weighting for better accuracy
            self.model = KNeighborsClassifier(
                n_neighbors=k,
                weights='distance'  # Closer neighbors matter more
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        self.save()

    def save(self):
        # Save both model and scaler
        data = {'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type}
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data.get('scaler', StandardScaler())
        self.model_type = data.get('model_type', 'knn')

    def predict(self, features):
        arr = np.array(features).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        return self.model.predict(arr_scaled)[0]
    
    def predict_with_confidence(self, features):
        """Return prediction and confidence score (0-1)"""
        arr = np.array(features).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        proba = self.model.predict_proba(arr_scaled)[0]
        pred = self.model.predict(arr_scaled)[0]
        confidence = np.max(proba)  # Highest probability = confidence
        return pred, confidence