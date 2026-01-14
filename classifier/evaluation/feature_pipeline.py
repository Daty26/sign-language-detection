import cv2
import numpy as np
from landmarks.mediapipe_wrapper import MediaPipeHandDetector
from landmarks.hand_landmarks import extract_first_hand
from features.feature_extractor import extract_features

class FeaturePipeline:
    def __init__(self):
        self.detector = MediaPipeHandDetector(
            max_num_hands=1,
            min_detection_confidence=0.02,
            min_tracking_confidence=0.05
        )

    def extract_from_bgr(self, frame_bgr):
        results = self.detector.process(frame_bgr)
        hand = extract_first_hand(results, frame_bgr.shape)
        if not hand:
            return None
        return extract_features(hand.landmarks)
    
    def ensure_min_size(self, img_bgr, min_side=320):
        h, w = img_bgr.shape[:2]
        m = min(h, w)
        if m >= min_side:
            return img_bgr
        scale = min_side / m
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def preprocess_for_detection(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

