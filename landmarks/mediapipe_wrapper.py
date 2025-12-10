import cv2
import mediapipe as mp
import numpy as np
from typing import Optional


class MediaPipeHandDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray):
        """
        Input: BGR-frame from OpenCV (frame.shape = (h, w, 3))
        Output: results от MediaPipe (results.multi_hand_landmarks, multi_handedness, ...)
        """
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        return results

    def close(self) -> None:
        self.hands.close()
