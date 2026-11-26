"""Latest-frame USB RGB camera reader, independent of robot and RealSense control."""

import threading
import time


class Webcam:
    def __init__(self, settings):
        self.settings = settings
        self.frame = None
        self.error = "Opening webcam"
        self.name = settings.get("name", "Webcam")
        self.stamp = 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self.thread.start()

    def _run(self):
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras

        cap = None
        try:
            desired = self.name.lower()
            devices = [
                c for c in enumerate_cameras(cv2.CAP_DSHOW) if desired in c.name.lower()
            ]
            if not devices:
                raise RuntimeError(f"{self.name} is not connected")
            camera = devices[0]
            self.name = camera.name
            cap = cv2.VideoCapture(camera.index, camera.backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 15)
            if not cap.isOpened():
                raise RuntimeError("Webcam is busy or unavailable")
            while not self.stop.is_set():
                ok, bgr = cap.read()
                if not ok:
                    self.error = "Webcam frame unavailable"
                    self.stop.wait(0.1)
                    continue
                self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.stamp, self.error = time.time(), None
        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    def read(self):
        if self.frame is None or time.time() - self.stamp > 2:
            return None
        return self.frame.copy(), self.stamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
