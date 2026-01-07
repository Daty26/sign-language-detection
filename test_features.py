import numpy as np
from features.geometry_utils import distance, angle_three_points
from features.feature_extractor import extract_features

def test_distance_zero():
    p = [0.0, 0.0, 0.0]
    assert distance(p, p) == 0.0

def test_angle_straight_line():
    a = [0, 0, 0]
    b = [1, 0, 0]
    c = [2, 0, 0]
    angle = angle_three_points(a, b, c)
    # straight line -> angle ≈ pi
    assert np.isclose(angle, np.pi, atol=1e-5)

def test_feature_length_extended():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    feats = extract_features(landmarks)
    assert feats.shape == (20,)

def test_feature_length_basic():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    feats = extract_features(landmarks, extended=False)
    assert feats.shape == (15,)
