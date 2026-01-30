#!/usr/bin/env python3
"""
Performance Tests - FPS Measurement
Tests: Real-time performance metrics

Author: Meirs (Integrator + Tester)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from camera.camera_stream import CameraStream


def test_camera_fps(duration=3):
    """Test camera capture FPS."""
    print("⚡ Test: Camera FPS")
    
    camera = None
    try:
        camera = CameraStream(width=416, height=416)
        camera.start()
        time.sleep(0.5)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            frame, meta = camera.read()
            frame_count += 1
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        camera.stop()
        
        print(f"  Camera FPS: {fps:.1f}")
        print(f"  Total frames: {frame_count}")
        print(f"  Resolution: 416x416")
        
        if fps >= 20:
            print(f"  ✅ Meets target (≥20 FPS)")
            return True
        else:
            print(f"  ⚠️ Below target: {fps:.1f} < 20")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        if camera:
            camera.stop()
        return False


def main():
    """Run performance tests."""
    print("="*60)
    print("⚡ PERFORMANCE TEST")
    print("="*60)
    print()
    
    result = test_camera_fps(duration=3)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    if result:
        print("✅ Camera optimized for real-time processing")
        print("Target: ≥20 FPS for smooth video capture")
    else:
        print("⚠️ Camera performance below target")
    print("="*60)
    
    return 0 if result else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
