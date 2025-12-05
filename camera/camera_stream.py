# camera_stream.py

import cv2
import time

class CameraStream:
    def __init__(self, width=640, height=480, target_fps=60):
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap = None
        self._last_time = None
        self.fps = 0.0

    # ---------- public API ----------
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        self._last_time = time.perf_counter()

    def read(self, crop_box=None):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame")

        frame = self._apply_crop(frame, crop_box)
        frame = self._resize(frame)
        frame = self._normalize(frame)
        frame = self._smooth(frame)
        self._update_fps()

        meta = {
            "fps": self.fps,
            "crop_box": crop_box,
            "size": (self.height, self.width),
        }
        self._sleep_to_match_fps()
        return frame, meta

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------- "private" helpers (internal use only) ----------
    def _apply_crop(self, frame, crop_box):
        if crop_box is None:
            return frame
        x1, y1, x2, y2 = crop_box
        return frame[y1:y2, x1:x2]

    def _resize(self, frame):
        return cv2.resize(frame, (self.width, self.height))

    def _normalize(self, frame):
        # BGR → RGB + scale to [0,1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype("float32") / 255.0

    def _smooth(self, frame):
        # simple example – can tweak later
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self._last_time if self._last_time is not None else 0
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self._last_time = now

    def _sleep_to_match_fps(self):
        # optional: simple FPS stabilizer
        if self.target_fps <= 0:
            return
        target_dt = 1.0 / self.target_fps
        now = time.perf_counter()
        dt = now - self._last_time
        extra = target_dt - dt
        if extra > 0:
            time.sleep(extra)
