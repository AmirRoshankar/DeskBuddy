import threading
import time

import pytest

from deskbuddy.latest_camera import LatestFrameCamera


def test_ui_read_returns_latest_while_capture_is_blocked_and_stale_is_rejected():
    release = threading.Event()
    blocked = threading.Event()
    ready = threading.Event()
    now = [0.0]
    frame = ("rgb", "depth", "intrinsics")

    class Camera:
        count = 0

        def read(self):
            self.count += 1
            if self.count == 1:
                ready.set()
                return frame
            blocked.set()
            release.wait(2)
            raise RuntimeError("Disconnected")

        def close(self):
            release.set()

    camera = LatestFrameCamera(Camera(), clock=lambda: now[0])
    try:
        assert ready.wait(1) and blocked.wait(1)
        started = time.monotonic()
        assert camera.read() == frame
        assert time.monotonic() - started < 0.05
        now[0] = 1.0
        with pytest.raises(RuntimeError):
            camera.read()
    finally:
        camera.close()
    assert not camera.thread.is_alive()
