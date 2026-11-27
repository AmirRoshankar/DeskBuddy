import time

import numpy as np
import pytest
from PIL import Image


class Root:
    def update_idletasks(self):
        pass

    def winfo_rootx(self):
        return 10

    def winfo_rooty(self):
        return 20

    def winfo_width(self):
        return 321

    def winfo_height(self):
        return 241

    def after(self, ms, callback):
        pass


def test_recording_writes_a_timestamped_real_time_mp4(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import PIL.ImageGrab

    from deskbuddy.ui.recorder import UiRecorder

    grabs = []

    def grab(bbox=None, all_screens=False):
        grabs.append(bbox)
        return Image.fromarray(np.full((241, 321, 3), len(grabs) % 255, np.uint8))

    monkeypatch.setattr(PIL.ImageGrab, "grab", grab)
    recorder = UiRecorder(Root(), fps=20, folder=tmp_path)
    path = recorder.start()
    assert path.name.startswith("deskbuddy-") and path.suffix == ".mp4"
    time.sleep(0.6)
    assert recorder.recording
    assert recorder.stop() == path and not recorder.recording
    assert recorder.error is None and grabs[0] == (10, 20, 331, 261)
    video = cv2.VideoCapture(str(path))
    assert video.isOpened()
    assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
    assert 6 <= int(video.get(cv2.CAP_PROP_FRAME_COUNT)) <= 16
