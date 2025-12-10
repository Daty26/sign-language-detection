from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class HandLandmarks:
    """
    landmarks: np.ndarray shape (21, 3), NORMALIZED coordinates.
    handedness: "Left" / "Right"
    score: confidence of model (0..1)
    """
    landmarks: np.ndarray
    handedness: str
    score: float


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalization:
    1) Transfer the coordinate system to the wrist (point 0)
    2) Scale so that the max distance across the palm is 1
    """
    # landmarks: (21, 3) 
    wrist = landmarks[0, :2].copy()  # x, y wrist
    landmarks[:, :2] -= wrist

    max_dist = np.linalg.norm(landmarks[:, :2], axis=1).max()
    if max_dist > 0:
        landmarks[:, :2] /= max_dist

    return landmarks


def extract_first_hand(
    results,
    image_shape: Tuple[int, int, int],
) -> Optional[HandLandmarks]:
    """
    Input:
      - results:  MediaPipeHandDetector.process()
      - image_shape: frame.shape (h, w, c)
    Output:
      - HandLandmarks, if the hand is found
      - None, hand not foud
    """
    if results is None or results.multi_hand_landmarks is None:
        return None

    h, w, _ = image_shape

    # We take the first hand we find
    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = lm.z  # relative value
        coords.append([x, y, z])

    coords = np.array(coords, dtype=np.float32)  # shape (21, 3)
    coords = _normalize_landmarks(coords)

    handedness = "Unknown"
    score = 0.0
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
        score = float(results.multi_handedness[0].classification[0].score)

    return HandLandmarks(landmarks=coords, handedness=handedness, score=score)


def landmarks_to_vector(hand: HandLandmarks) -> np.ndarray:
    """
    Input: HandLandmarks
    Output: 1D-vector of length 63 (21 points * 3 coordinates)
    """
    return hand.landmarks.flatten()
