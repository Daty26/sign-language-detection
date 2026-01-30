#!/usr/bin/env python3
"""
Sign Language Detection - Main Application
Integrates all modules: UI, camera, landmarks, features, and classifier.
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Import UI
from ui.ui_main import SignLanguageApp

# Import camera module
from camera.camera_stream import CameraStream

# Import detection & processing modules
from landmarks.mediapipe_wrapper import MediaPipeHandDetector
from landmarks.hand_landmarks import extract_first_hand
from features.feature_extractor import extract_features
from classifier.gesture_classifier import GestureClassifier


class SignLanguageDetectionApp(SignLanguageApp):
    """Extended app with full gesture detection pipeline."""
    
    def __init__(self):
        super().__init__()
        
        # Replace cv2.VideoCapture with CameraStream
        self.camera = None
        
        # Initialize detection pipeline with optimized settings
        self.detector = MediaPipeHandDetector(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.classifier = GestureClassifier()
        
        # Gesture tracking
        self.last_gesture = None
        self.gesture_counter = 0
        
        # Performance optimization: frame skipping
        self.frame_count = 0
        self.process_every_n_frames = 2  # Process every 2nd frame
        self.last_results = None  # Cache last detection results
    
    def start_detection(self):
        """Override to use CameraStream instead of cv2.VideoCapture."""
        if not self.is_running:
            self.is_running = True
            # Use smaller resolution for better FPS (416x416 good for MediaPipe)
            self.camera = CameraStream(width=416, height=416)
            try:
                self.camera.start()
                self.update_loop()
            except RuntimeError as e:
                print(f"Could not start camera: {e}")
                self.is_running = False
                self.camera = None
    
    def stop_detection(self):
        """Override to use CameraStream."""
        self.is_running = False
        if self.camera:
            self.camera.stop()
            self.camera = None
        
        self.video_frame.label.configure(image='', text="Camera Paused")
        self.last_gesture = None
        self.gesture_counter = 0
        self.frame_count = 0
        self.last_results = None
        
    def update_loop(self):
        """Override update_loop to add gesture detection with optimizations."""
        if self.is_running and self.camera:
            try:
                # Read frame from CameraStream
                frame_normalized, meta = self.camera.read()
                frame_bgr = (frame_normalized * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                
                # Performance optimization: process detection every N frames
                self.frame_count += 1
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                
                if should_process:
                    # Process frame with full pipeline
                    results = self.detector.process(frame_bgr)
                    self.last_results = results  # Cache for next frames
                else:
                    # Reuse cached results for smoother display
                    results = self.last_results
                
                # Extract hand and process gesture (even with cached results)
                if results:
                    hand = extract_first_hand(results, frame_bgr.shape)
                    
                    # Detect gesture if hand found
                    if hand:
                        # Extract features and classify
                        features_vector = extract_features(hand.landmarks)
                        features_dict = self._features_to_dict(features_vector)
                        gesture = self.classifier.classify(features_dict, features_vector)
                        
                        # Add to output if gesture changed and stable
                        if gesture != "Unknown" and gesture != self.last_gesture:
                            self.gesture_counter += 1
                            if self.gesture_counter > 3:
                                self.text_box.add_text(gesture)
                                self.last_gesture = gesture
                                self.gesture_counter = 0
                        elif gesture == self.last_gesture:
                            self.gesture_counter = 0
                        
                        # Draw landmarks using MediaPipe drawing utils
                        if results.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame_bgr,
                                results.multi_hand_landmarks[0],
                                mp.solutions.hands.HAND_CONNECTIONS
                            )
                    else:
                        self.last_gesture = None
                        self.gesture_counter = 0
                
                # Display frame (always, for smooth video)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                viewport_width = self.video_frame.winfo_width()
                viewport_height = self.video_frame.winfo_height()
                if viewport_width < 10 or viewport_height < 10:
                    viewport_width, viewport_height = 640, 480
                pil_image = pil_image.resize((viewport_width, viewport_height))
                self.video_frame.update_image(pil_image)
                
            except Exception as e:
                print(f"Frame processing error: {e}")
        
        # Reduced delay for faster loop (5ms instead of 10ms)
        self.after(5, self.update_loop)
    
    def _features_to_dict(self, features_vector):
        """Convert feature vector to dict for rule-based classifier."""
        flags = features_vector[5:10]  # finger up/down flags
        palm_normal = features_vector[12:15]  # palm orientation vector
        
        # Determine palm orientation from normal vector
        palm_orientation = "front"
        if abs(palm_normal[2]) > 0.7:
            palm_orientation = "front" if palm_normal[2] > 0 else "back"
        elif abs(palm_normal[1]) > 0.5:
            palm_orientation = "down" if palm_normal[1] > 0 else "up"
        else:
            palm_orientation = "side"
        
        return {
            "thumb": int(flags[0]),
            "index": int(flags[1]),
            "middle": int(flags[2]),
            "ring": int(flags[3]),
            "pinky": int(flags[4]),
            "palm_orientation": palm_orientation,
            "hand_movement": "none"
        }
    


if __name__ == "__main__":
    app = SignLanguageDetectionApp()
    app.mainloop()