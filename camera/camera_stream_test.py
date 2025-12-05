# camera_stream_test.py
import cv2
from camera_stream import CameraStream

def main():
    stream = CameraStream()
    stream.start()

    print("Press 's' to save a frame, 'q' to quit.")

    while True:
        frame, meta = stream.read()

        # Convert normalized RGB float32 → BGR uint8 for display & saving
        disp = (frame * 255).astype("uint8")
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

        # Show FPS on screen
        cv2.putText(
            disp_bgr, f"FPS: {meta['fps']:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("Camera Stream Test", disp_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save frame
            filename = "saved_frame.jpg"
            cv2.imwrite(filename, disp_bgr)
            print(f"Saved frame to: {filename}")

        elif key == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
