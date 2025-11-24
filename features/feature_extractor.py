# Feature vector layout:
# [0:5]  finger angles   [thumb, index, middle, ring, pinky]  (normalized by pi)
# [5:10] finger flags    [thumb, index, middle, ring, pinky]  (0=down, 1=up)
# [10]   d_thumb_index   (normalized by palm size)
# [11]   d_index_middle  (normalized by palm size)
# [12:15] palm normal    (x, y, z)
import numpy as np
from typing import Dict, List

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
    Returns a 3D vector (not necessarily unit).
    """
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    v1 = vector(wrist, index_mcp)
    v2 = vector(wrist, pinky_mcp)
    normal = np.cross(v1, v2)
    normal = normalize_vector(normal)
    return normal  # shape (3,)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert 21x3 hand landmarks into a compact feature vector.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (21, 3) with MediaPipe hand landmarks
        in normalized coordinates.

    Returns
    -------
    np.ndarray
        1D feature vector containing:
        - 5 normalized finger angles (0..1)
        - 5 finger up/down flags (0 or 1)
        - 2 normalized distances
        - 3D palm normal vector
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

    # 1) Finger angles
    finger_angles = compute_finger_angles(landmarks)
    # Convert radians to something normalized [0, 1] (divide by pi)
    angle_features = np.array(
        [finger_angles[f] for f in ["thumb", "index", "middle", "ring", "pinky"]],
        dtype=np.float32,
    ) / np.pi

    # 2) Finger up flags (0/1)
    finger_flags = compute_finger_up_flags(landmarks)
    flag_features = np.array(
        [finger_flags[f] for f in ["thumb", "index", "middle", "ring", "pinky"]],
        dtype=np.float32,
    )

    # 3) Distances, normalized by palm size
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    palm_size = distance(wrist, middle_mcp) + 1e-6  # avoid division by zero

    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]

    d_thumb_index = distance(thumb_tip, index_tip) / palm_size
    d_index_middle = distance(index_tip, middle_tip) / palm_size

    distance_features = np.array(
        [d_thumb_index, d_index_middle], dtype=np.float32
    )

    # 4) Palm orientation vector
    palm_normal = compute_palm_orientation(landmarks)  # (3,)
    palm_features = palm_normal.astype(np.float32)

    # Concatenate everything into one feature vector
    feature_vector = np.concatenate(
        [angle_features, flag_features, distance_features, palm_features], axis=0
    )

    return feature_vector
