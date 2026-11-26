"""Non-blocking latest camera frame for Tk; no SDK waits on the UI thread."""

import threading
import time


class LatestFrameCamera:
    def __init__(self, camera, max_age_s=0.5, clock=time.monotonic):
        self.camera, self.max_age_s, self.clock = camera, max_age_s, clock
        self.sample = None
        self.error = "Waiting for first camera frame"
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._capture, name="realsense-capture", daemon=True
        )
        self.thread.start()

    def _capture(self):
        while not self.stop.is_set():
            began = time.monotonic()
            try:
                frame = self.camera.read()
                self.sample = (frame, self.clock())
                self.error = None
            except (OSError, RuntimeError, ValueError) as exc:
                self.error = str(exc)
            # Avoid spinning on a disconnected SDK or a fast synthetic source.
            self.stop.wait(max(0.0, 0.01 - (time.monotonic() - began)))

    def read(self):
        sample = self.sample
        if sample is None or self.clock() - sample[1] > self.max_age_s:
            raise RuntimeError(self.error or "Camera frame is stale")
        return sample[0]

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def close(self):
        self.stop.set()
        try:
            self.camera.close()
        finally:
            self.thread.join(timeout=1.2)
