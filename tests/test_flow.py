#!/usr/bin/env python3
"""
Integration Tests - Pipeline Flow
Tests: camera → landmarks → features → classifier

Author: Meirs (Integrator + Tester)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
from camera.camera_stream import CameraStream
from landmarks.mediapipe_wrapper import MediaPipeHandDetector
from landmarks.hand_landmarks import extract_first_hand
from features.feature_extractor import extract_features
from classifier.gesture_classifier import GestureClassifier


def test_module_initialization():
    """Test: All modules initialize correctly."""
    print("🧪 Test 1: Module Initialization")
    
    try:
        camera = CameraStream()
        detector = MediaPipeHandDetector()
        classifier = GestureClassifier()
        print("✅ All modules initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def test_camera_to_detector():
    """Test: Camera → Detector integration."""
    print("\n🧪 Test 2: Camera → Detector Integration")
    
    camera = None
    try:
        camera = CameraStream(width=416, height=416)
        camera.start()
        time.sleep(0.5)
        
        frame_normalized, meta = camera.read()
        frame_bgr = (frame_normalized * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        
        detector = MediaPipeHandDetector()
        results = detector.process(frame_bgr)
        
        camera.stop()
        print("✅ Camera → Detector integration working")
        return True
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        if camera:
            camera.stop()
        return False


def test_full_pipeline():
    """Test: Full pipeline (camera → detector → features → classifier)."""
    print("\n🧪 Test 3: Full Pipeline Integration")
    
    camera = None
    try:
        camera = CameraStream(width=416, height=416)
        camera.start()
        time.sleep(0.5)
        
        # Read frame
        frame_normalized, meta = camera.read()
        frame_bgr = (frame_normalized * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        
        # Detect hand
        detector = MediaPipeHandDetector()
        results = detector.process(frame_bgr)
        hand = extract_first_hand(results, frame_bgr.shape)
        
        # If hand found, test features and classifier
        if hand:
            features_vector = extract_features(hand.landmarks)
            
            # Convert to dict for classifier
            flags = features_vector[5:10]
            features_dict = {
                "thumb": int(flags[0]),
                "index": int(flags[1]),
                "middle": int(flags[2]),
                "ring": int(flags[3]),
                "pinky": int(flags[4]),
                "palm_orientation": "front",
                "hand_movement": "none"
            }
            
            classifier = GestureClassifier()
            gesture = classifier.classify(features_dict, features_vector)
            
            print(f"✅ Full pipeline working (detected: {gesture})")
        else:
            print("✅ Full pipeline working (no hand detected)")
        
        camera.stop()
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        if camera:
            camera.stop()
        return False


def test_multi_frame_stability():
    """Test: Multi-frame processing stability."""
    print("\n🧪 Test 4: Multi-frame Stability")
    
    camera = None
    try:
        camera = CameraStream(width=416, height=416)
        camera.start()
        time.sleep(0.5)
        
        detector = MediaPipeHandDetector()
        
        # Process 10 frames
        for i in range(10):
            frame_normalized, meta = camera.read()
            frame_bgr = (frame_normalized * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            results = detector.process(frame_bgr)
            time.sleep(0.03)
        
        camera.stop()
        print("✅ Multi-frame processing stable")
        return True
        
    except Exception as e:
        print(f"❌ Stability test failed: {e}")
        if camera:
            camera.stop()
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("🔬 INTEGRATION TESTS")
    print("="*60)
    print()
    
    results = []
    results.append(test_module_initialization())
    results.append(test_camera_to_detector())
    results.append(test_full_pipeline())
    results.append(test_multi_frame_stability())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Tests: {passed}/{total} passed")
    print(f"Pass rate: {passed/total*100:.0f}%")
    print("="*60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
