import numpy as np

from features.feature_extractor import extract_features


def main():
    # Fake landmarks: 21 points, 3 coordinates (x, y, z)
    # In the real app this comes from MediaPipe, but for testing we just use random numbers
    landmarks = np.random.rand(21, 3).astype(np.float32)

    features = extract_features(landmarks)

    print("Feature vector shape:", features.shape)
    print("Feature vector:", features)

    # Optional: print in sections so you see what is what
    print("\n--- Parsed features ---")
    print("Finger angles   [0:5]  :", features[0:5])
    print("Finger flags    [5:10] :", features[5:10])
    print("Distances       [10:12]:", features[10:12])
    print("Palm normal     [12:15]:", features[12:15])
    print("Hand openness   [15]   :", features[15])
    print("Finger spread   [16:20]:", features[16:20])


if __name__ == "__main__":
    main()
