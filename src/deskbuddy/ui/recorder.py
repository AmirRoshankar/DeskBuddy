"""Record the whole DeskBuddy window (cameras, 3D view, logs) to an MP4.

Screen capture of the window's on-screen rectangle, so everything visible is
recorded exactly as shown. Frames are grabbed in a background thread at a
fixed rate (the last frame is repeated if a grab is slow, keeping real time)
and written to runs/recordings/deskbuddy-YYYYmmdd-HHMMSS.mp4. Presentation
only: no robot or camera API is touched.
"""

import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path("runs/recordings")


class UiRecorder:
    def __init__(self, root, fps=10, folder=ROOT):
        self.root, self.fps, self.folder = root, fps, Path(folder)
        self.thread, self.stop_event = None, threading.Event()
        self.bbox, self.path, self.frames, self.error = None, None, 0, None
        self.started = None

    @property
    def recording(self):
        return self.thread is not None and self.thread.is_alive()

    def _track_window(self):
        # Tk geometry must be read on the UI thread; the capture thread only
        # reads the latest rectangle.
        if self.thread is None:
            return
        try:
            self.root.update_idletasks()
            x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
            w, h = self.root.winfo_width(), self.root.winfo_height()
            if w > 50 and h > 50:
                self.bbox = (x, y, x + w, y + h)
        except Exception:  # noqa: BLE001 - window closing
            return
        self.root.after(500, self._track_window)

    def start(self):
        if self.recording:
            return self.path
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"deskbuddy-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        self.stop_event.clear()
        self.frames, self.error, self.started = 0, None, time.monotonic()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="ui-recorder"
        )
        self._track_window()
        self.thread.start()
        return self.path

    def stop(self):
        if self.thread is None:
            return None
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.thread = None
        return self.path

    def elapsed_s(self):
        return 0.0 if self.started is None else time.monotonic() - self.started

    def _run(self):
        import cv2
        from PIL import ImageGrab

        writer, size, last = None, None, None
        period = 1.0 / self.fps
        next_time = time.monotonic()
        try:
            while not self.stop_event.is_set():
                bbox = self.bbox
                if bbox is None:
                    time.sleep(0.05)
                    continue
                try:
                    image = ImageGrab.grab(bbox=bbox, all_screens=True)
                    frame = cv2.cvtColor(
                        np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR
                    )
                except OSError:
                    frame = last
                if frame is None:
                    continue
                if writer is None:
                    # Even dimensions keep common MP4 players happy.
                    size = (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2)
                    writer = cv2.VideoWriter(
                        str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
                    )
                    if not writer.isOpened():
                        raise OSError("Video writer could not open the MP4 file")
                if (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size)
                last = frame
                now = time.monotonic()
                # Real-time timeline: repeat the frame for any missed slots.
                while next_time <= now:
                    writer.write(frame)
                    self.frames += 1
                    next_time += period
                time.sleep(max(0.0, next_time - time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()
