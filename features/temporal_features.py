import numpy as np

def compute_motion_features(
    prev_landmarks: np.ndarray,
    curr_landmarks: np.ndarray
) -> np.ndarray:
    """
    Simple motion features between two frames.
    - Mean speed of all landmarks
    - Max speed of any landmark
    - Speed of wrist
    """
    prev = prev_landmarks.astype(np.float32)
    curr = curr_landmarks.astype(np.float32)

    diffs = curr - prev            # (21, 3)
    speeds = np.linalg.norm(diffs, axis=1)  # (21,)

    mean_speed = speeds.mean()
    max_speed = speeds.max()
    wrist_speed = speeds[0]  # WRIST index = 0

    return np.array([mean_speed, max_speed, wrist_speed], dtype=np.float32)
