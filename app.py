"""
Sign Language Detection - Main Application
Integrates all modules: camera → landmarks → features → classifier → UI
"""

import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Dict, Tuple
from PIL import Image

import customtkinter as ctk

from camera.camera_stream import CameraStream
from landmarks import MediaPipeHandDetector, extract_first_hand
from features import extract_features
from classifier.gesture_classifier import GestureClassifier
from ui.ui_components import VideoFrame, TextOutputBox, ControlPanel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set appearance mode for customtkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class FeatureAdapter:
    """
    Adapter to convert feature vector (numpy array) to dictionary format
    for RuleBasedClassifier, and determine palm orientation and hand movement.
    """
    
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
    
    def feature_vector_to_dict(self, feature_vector: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Convert feature vector to dictionary format for RuleBasedClassifier.
        
        Feature vector layout:
        [0:5]  finger angles   [thumb, index, middle, ring, pinky]
        [5:10] finger flags    [thumb, index, middle, ring, pinky]  (0=down, 1=up)
        [10]   d_thumb_index
        [11]   d_index_middle
        [12:15] palm normal    (x, y, z)
        """
        # Extract finger flags (already 0/1)
        finger_flags = feature_vector[5:10]
        
        # Determine palm orientation from palm normal (z-component)
        palm_normal = feature_vector[12:15]
        palm_z = palm_normal[2]  # z-component
        
        # Simple heuristic: z > 0.5 = front, z < -0.5 = down, else = side
        if palm_z > 0.5:
            palm_orientation = "front"
        elif palm_z < -0.5:
            palm_orientation = "down"
        else:
            palm_orientation = "side"
        
        # Determine hand movement from position history
        hand_movement = "none"
        if landmarks is not None:
            # Use wrist position as reference
            wrist_pos = landmarks[0, :2]  # x, y
            self.position_history.append(wrist_pos)
            
            if len(self.position_history) >= 2:
                # Calculate average movement direction
                positions = np.array(list(self.position_history))
                if len(positions) >= 2:
                    # Movement in y-direction (forward = negative y in image coords)
                    y_movement = positions[-1][1] - positions[0][1]
                    # Threshold for forward movement
                    if y_movement < -0.05:  # Moving up in image = forward
                        hand_movement = "forward"
        
        return {
            "thumb": int(finger_flags[0]),
            "index": int(finger_flags[1]),
            "middle": int(finger_flags[2]),
            "ring": int(finger_flags[3]),
            "pinky": int(finger_flags[4]),
            "palm_orientation": palm_orientation,
            "hand_movement": hand_movement,
        }


class PredictionSmoother:
    """
    Smooths predictions using a sliding window to reduce flickering.
    """
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
    
    def add_prediction(self, prediction: str):
        """Add a new prediction to the history."""
        if prediction and prediction != "Unknown":
            self.prediction_history.append(prediction)
    
    def get_smoothed_prediction(self) -> Optional[str]:
        """
        Return the most common prediction in the window.
        If window is not full, return the most recent.
        """
        if not self.prediction_history:
            return None
        
        if len(self.prediction_history) < self.window_size:
            return self.prediction_history[-1]
        
        # Count occurrences
        from collections import Counter
        counter = Counter(self.prediction_history)
        return counter.most_common(1)[0][0]


class SignLanguageApp(ctk.CTk):
    """
    Main application class that integrates all modules.
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("Sign Language Translator AI")
        self.geometry("1100x700")
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Pipeline components
        self.camera = None
        self.detector = None
        self.classifier = None
        self.feature_adapter = None
        self.smoother = None
        
        # State
        self.is_running = False
        self.last_prediction = None
        self.frame_counter = 0
        self.process_every_n_frames = 4  # Process every 4th frame for detection (very aggressive optimization for FPS)
        self.last_viewport_size = (640, 480)
        self.fps_update_counter = 0
        self.fps_update_interval = 15  # Update FPS label every 15 frames
        self.last_pil_image = None  # Cache last image to avoid unnecessary conversions
        
        # UI Components
        self.video_frame = VideoFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.right_sidebar = ctk.CTkFrame(self, fg_color="transparent")
        self.right_sidebar.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.controls = ControlPanel(
            self.right_sidebar,
            start_cb=self.start_detection,
            stop_cb=self.stop_detection,
            clear_cb=self.clear_log,
            save_cb=self.save_log
        )
        self.controls.pack(side="bottom", fill="x", pady=10)
        
        self.text_box = TextOutputBox(self.right_sidebar)
        self.text_box.pack(side="top", fill="both", expand=True)
        
        # FPS label
        self.fps_label = ctk.CTkLabel(
            self.right_sidebar,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12)
        )
        self.fps_label.pack(side="top", pady=(0, 10))
        
    def initialize_pipeline(self):
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing camera...")
            # Reduced resolution for better FPS (320x240 is very fast)
            self.camera = CameraStream(width=320, height=240, target_fps=30)  # Lower resolution = much better FPS
            self.camera.start()
            
            logger.info("Initializing MediaPipe hand detector...")
            # Lower confidence thresholds for faster processing
            self.detector = MediaPipeHandDetector(
                max_num_hands=1,
                min_detection_confidence=0.5,  # Lower = faster
                min_tracking_confidence=0.4  # Lower = faster
            )
            
            logger.info("Initializing gesture classifier...")
            self.classifier = GestureClassifier()
            
            logger.info("Initializing feature adapter and smoother...")
            self.feature_adapter = FeatureAdapter(history_size=5)
            self.smoother = PredictionSmoother(window_size=5)
            
            logger.info("Pipeline initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            return False
    
    def process_frame(self, frame_rgb: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray], np.ndarray]:
        """
        Process a single frame through the pipeline - optimized for speed.
        
        Returns:
            (prediction, landmarks_array, display_frame) or (None, None, display_frame)
        """
        try:
            # Convert RGB float32 to BGR uint8 for MediaPipe (optimized - single conversion)
            frame_uint8 = (frame_rgb * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            
            # Create display frame (reuse conversion)
            display_frame = frame_uint8.copy()
            
            # Detect hand - this is the slowest operation
            results = self.detector.process(frame_bgr)
            hand = extract_first_hand(results, frame_bgr.shape)
            
            if hand is None:
                return None, None, display_frame
            
            # Extract features
            feature_vector = extract_features(hand.landmarks)
            
            # Convert to dictionary format for rule-based classifier
            feature_dict = self.feature_adapter.feature_vector_to_dict(
                feature_vector, hand.landmarks
            )
            
            # Classify
            prediction = self.classifier.classify(
                features=feature_dict,
                ml_vector=feature_vector
            )
            
            # Draw prediction on frame (simplified drawing for speed)
            if prediction and prediction != "Unknown":
                cv2.putText(
                    display_frame,
                    f"Gesture: {prediction}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # Smaller font = faster
                    (0, 255, 0),
                    2
                )
            
            return prediction, hand.landmarks, display_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            display_frame = (frame_rgb * 255).astype(np.uint8)
            return None, None, display_frame
    
    def start_detection(self):
        """Start the detection pipeline."""
        if not self.is_running:
            if not self.initialize_pipeline():
                logger.error("Failed to initialize pipeline")
                return
            
            self.is_running = True
            logger.info("Detection started")
            self.update_loop()
    
    def stop_detection(self):
        """Stop the detection pipeline."""
        self.is_running = False
        
        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
        
        if self.detector:
            try:
                self.detector.close()
            except Exception as e:
                logger.error(f"Error closing detector: {e}")
        
        self.video_frame.label.configure(image='', text="Camera Paused")
        logger.info("Detection stopped")
    
    def clear_log(self):
        """Clear the text output box."""
        self.text_box.clear()
    
    def save_log(self):
        """Save the log to a file."""
        import datetime
        text = self.text_box.get_content()
        filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Log saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving log: {e}")
    
    def update_loop(self):
        """Main update loop for processing frames - optimized for 20-30 FPS."""
        if not self.is_running:
            return
        
        try:
            # Read frame from camera
            frame_rgb, meta = self.camera.read()
            fps = meta.get("fps", 0.0)
            self.frame_counter += 1
            
            # Update FPS label less frequently (optimization)
            self.fps_update_counter += 1
            if self.fps_update_counter >= self.fps_update_interval:
                self.fps_label.configure(text=f"FPS: {fps:.1f}")
                self.fps_update_counter = 0
            
            # Process frame through pipeline (skip more frames for optimization)
            prediction = None
            display_frame = None
            
            if self.frame_counter % self.process_every_n_frames == 0:
                # Full processing: detection + classification
                prediction, landmarks, display_frame = self.process_frame(frame_rgb)
                
                # Handle prediction
                if prediction and prediction != "Unknown":
                    # Add to smoother
                    self.smoother.add_prediction(prediction)
                    smoothed = self.smoother.get_smoothed_prediction()
                    
                    # Only add to text box if prediction changed
                    if smoothed and smoothed != self.last_prediction:
                        self.text_box.add_text(smoothed)
                        self.last_prediction = smoothed
                        # Log less frequently for performance
                        if self.frame_counter % 30 == 0:
                            logger.info(f"Recognized gesture: {smoothed}")
            else:
                # Fast path: just get display frame without processing
                display_frame = (frame_rgb * 255).astype(np.uint8)
                # Draw last prediction if available (simplified)
                if self.last_prediction:
                    cv2.putText(
                        display_frame,
                        f"Gesture: {self.last_prediction}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
            
            # Optimize PIL conversion - only convert if needed
            if display_frame is not None:
                # Convert display frame to PIL Image for UI (cache if possible)
                pil_image = Image.fromarray(display_frame)
                
                # Get viewport dimensions (cache very aggressively - check every 10 frames)
                if self.frame_counter % 10 == 0:  # Check viewport size every 10 frames
                    viewport_width = self.video_frame.winfo_width()
                    viewport_height = self.video_frame.winfo_height()
                    
                    if viewport_width < 10 or viewport_height < 10:
                        viewport_width, viewport_height = self.last_viewport_size
                    else:
                        self.last_viewport_size = (viewport_width, viewport_height)
                else:
                    viewport_width, viewport_height = self.last_viewport_size
                
                # Resize only if necessary and cache result
                if (viewport_width, viewport_height) != pil_image.size:
                    # Limit max size to avoid huge resize operations
                    max_width, max_height = 640, 480
                    if viewport_width > max_width:
                        viewport_width = max_width
                    if viewport_height > max_height:
                        viewport_height = max_height
                    
                    pil_image = pil_image.resize(
                        (viewport_width, viewport_height), 
                        Image.Resampling.NEAREST  # Fastest resampling
                    )
                
                # Update video frame
                self.video_frame.update_image(pil_image)
            
        except Exception as e:
            logger.error(f"Error in update loop: {e}", exc_info=True)
            self.stop_detection()
            return
        
        # Schedule next update - optimized for 20-30 FPS (33ms = ~30 FPS)
        self.after(33, self.update_loop)  # 1000ms / 30 FPS = 33ms
    
    def on_closing(self):
        """Handle window closing."""
        self.stop_detection()
        self.destroy()


def main():
    """Entry point."""
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()

