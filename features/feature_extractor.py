# Feature vector layout (length = 20):
# [0:5]   finger angles   [thumb, index, middle, ring, pinky]  (normalized by pi)
# [5:10]  finger flags    [thumb, index, middle, ring, pinky]  (0=down, 1=up)
# [10]    d_thumb_index   (normalized by palm size / in normalized space)
# [11]    d_index_middle  (normalized by palm size / in normalized space)
# [12:15] palm normal     (x, y, z)
# [15]    hand openness   (average fingertip distance from palm center)
# [16:20] finger spread   (thumb–index, index–middle, middle–ring, ring–pinky)

import numpy as np
from typing import Dict, List, Sequence, Tuple

from .geometry_utils import (
    distance,
    angle_three_points,
    vector,
    normalize_vector,
)

# MediaPipe Hands indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def _finger_joint_sets() -> Dict[str, List[int]]:
    """
    Returns landmark indices for each finger: [MCP, PIP, TIP]
    (Thumb is a bit special, but we approximate similarly.)
    """
    return {
        "thumb": [THUMB_MCP, THUMB_IP, THUMB_TIP],
        "index": [INDEX_MCP, INDEX_PIP, INDEX_TIP],
        "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP],
        "ring": [RING_MCP, RING_PIP, RING_TIP],
        "pinky": [PINKY_MCP, PINKY_PIP, PINKY_TIP],
    }


def normalize_landmarks(landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Translate and scale landmarks so that:
    - Wrist is at (0, 0, 0)
    - Palm size (wrist -> middle MCP) is ~1

    Returns
    -------
    landmarks_norm : np.ndarray
        Normalized landmarks of shape (21, 3).
    palm_size : float
        Original palm size used for scaling.
    """
    wrist = landmarks[WRIST]
    shifted = landmarks - wrist

    middle_mcp = landmarks[MIDDLE_MCP]
    palm_size = distance(wrist, middle_mcp) + 1e-6  # avoid division by zero

    normalized = shifted / palm_size
    return normalized, palm_size


def compute_finger_angles(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Compute flexion angle for each finger in radians.

    landmarks: (21, 3)
    returns: dict {finger_name: angle}
    """
    angles = {}
    joints = _finger_joint_sets()

    for name, (mcp, pip, tip) in joints.items():
        a = landmarks[mcp]
        b = landmarks[pip]
        c = landmarks[tip]
        angle = angle_three_points(a, b, c)  # angle at PIP
        angles[name] = angle
    return angles


def compute_finger_up_flags(landmarks: np.ndarray) -> Dict[str, int]:
    """
    Very simple heuristic: finger is 'up' if fingertip is above PIP
    in image coordinates (assuming y increases downwards).
    """
    flags = {}
    joints = _finger_joint_sets()

    for name, (_, pip, tip) in joints.items():
        pip_y = landmarks[pip, 1]
        tip_y = landmarks[tip, 1]
        flags[name] = int(tip_y < pip_y)  # tip higher => finger up
    return flags


def compute_palm_orientation(landmarks: np.ndarray) -> np.ndarray:
    """
    Estimate palm normal using wrist, index_mcp, pinky_mcp.
    Returns a unit 3D vector.
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    v1 = vector(wrist, index_mcp)
    v2 = vector(wrist, pinky_mcp)
    normal = np.cross(v1, v2)
    normal = normalize_vector(normal)
    return normal  # shape (3,)


def compute_hand_openness(landmarks_norm: np.ndarray) -> float:
    """
    Measure how 'open' the hand is: average distance of fingertips from palm center.
    Uses normalized landmarks (palm size ~ 1).
    """
    fingertip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    tips = landmarks_norm[fingertip_indices]  # (5, 3)
    palm_center = landmarks_norm[WRIST]       # after normalization this is (0,0,0)
    dists = np.linalg.norm(tips - palm_center, axis=1)
    return float(dists.mean())


def compute_finger_spread(landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Distances between adjacent fingertips in normalized space.
    Order: thumb-index, index-middle, middle-ring, ring-pinky
    """
    indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    tips = landmarks_norm[indices]

    spreads = []
    for i in range(len(tips) - 1):
        spreads.append(np.linalg.norm(tips[i + 1] - tips[i]))

    return np.asarray(spreads, dtype=np.float32)  # shape (4,)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert 21x3 hand landmarks into a compact feature vector.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (21, 3) with MediaPipe hand landmarks
        in normalized coordinates (as given by the pipeline).

    Returns
    -------
    np.ndarray
        1D feature vector containing:
        - 5 normalized finger angles (0..1)
        - 5 finger up/down flags (0 or 1)
        - 2 distances between fingertips (in normalized space)
        - 3D palm normal vector
        - 1 hand openness scalar
        - 4 finger spread distances (adjacent fingertips)
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

    # Normalize landmarks for scale & translation invariance
    landmarks_norm, palm_size = normalize_landmarks(landmarks)

    # 1) Finger angles (use normalized landmarks; angles invariant to translation/scale)
    finger_angles = compute_finger_angles(landmarks_norm)
    # Convert radians to [0, 1] (divide by pi)
    angle_features = np.array(
        [finger_angles[f] for f in ["thumb", "index", "middle", "ring", "pinky"]],
        dtype=np.float32,
    ) / np.pi

    # 2) Finger up flags (0/1) - y-relations preserved under our normalization
    finger_flags = compute_finger_up_flags(landmarks_norm)
    flag_features = np.array(
        [finger_flags[f] for f in ["thumb", "index", "middle", "ring", "pinky"]],
        dtype=np.float32,
    )

    # 3) Distances between fingertips in normalized space
    thumb_tip = landmarks_norm[THUMB_TIP]
    index_tip = landmarks_norm[INDEX_TIP]
    middle_tip = landmarks_norm[MIDDLE_TIP]

    d_thumb_index = distance(thumb_tip, index_tip)       # already scale-normalized
    d_index_middle = distance(index_tip, middle_tip)

    distance_features = np.array(
        [d_thumb_index, d_index_middle], dtype=np.float32
    )

    # 4) Palm orientation vector (uses normalized landmarks; direction is what matters)
    palm_normal = compute_palm_orientation(landmarks_norm)  # (3,)
    palm_features = palm_normal.astype(np.float32)

    # 5) Hand openness
    openness = np.array([compute_hand_openness(landmarks_norm)], dtype=np.float32)

    # 6) Finger spread features
    spread_features = compute_finger_spread(landmarks_norm)  # (4,)

    # Concatenate everything into one feature vector
    feature_vector = np.concatenate(
        [
            angle_features,
            flag_features,
            distance_features,
            palm_features,
            openness,
            spread_features,
        ],
        axis=0,
    )

    return feature_vector
