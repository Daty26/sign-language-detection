import numpy as np
from typing import Sequence

Point3D = Sequence[float]


def to_np(p: Point3D) -> np.ndarray:
    return np.asarray(p, dtype=np.float32)


def distance(p1: Point3D, p2: Point3D) -> float:
    """Euclidean distance between two 3D points."""
    p1, p2 = to_np(p1), to_np(p2)
    return float(np.linalg.norm(p2 - p1))


def vector(p1: Point3D, p2: Point3D) -> np.ndarray:
    """Vector from p1 to p2."""
    p1, p2 = to_np(p1), to_np(p2)
    return p2 - p1


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return unit vector, or v if norm is zero."""
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between vectors v1 and v2 in radians.
    """
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.arccos(dot))


def angle_three_points(a: Point3D, b: Point3D, c: Point3D) -> float:
    """
    Angle ABC (at point b) in radians.
    a, b, c are 3D points.
    """
    ba = vector(b, a)
    bc = vector(b, c)
    return angle_between(ba, bc)
