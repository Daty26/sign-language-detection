import numpy as np
from features.feature_extractor import extract_features

# Fake landmarks: 21 points, 3 coords each
landmarks = np.random.rand(21, 3).astype(np.float32)

fv = extract_features(landmarks)
print("Feature vector shape:", fv.shape)
print("Feature vector:", fv)
