from types import SimpleNamespace

from deskbuddy.ui.paired_calibration import PairedCalibration


def test_eight_pairs_require_sixteen_clicks_and_close_automatically():
    view = PairedCalibration.__new__(PairedCalibration)
    view.count, view.pixels, view.context = 0, [None, None], None
    view.labels = [
        SimpleNamespace(viewport=SimpleNamespace(to_camera=lambda x, y: (x, y)))
        for _ in range(2)
    ]
    view.capture_state = lambda: {"q_model_rad": [0, 0, 0, 0, 0]}
    view.status = SimpleNamespace(set=lambda text: None)
    view.images = [None, None]
    saved, closed = [], []
    view.save_pair = lambda pixels, context, images: saved.append(pixels)
    view.close = lambda: closed.append(True)
    for i in range(8):
        view.click(0, SimpleNamespace(x=10, y=20))
        assert len(saved) == i
        view.click(1, SimpleNamespace(x=30, y=40))
        assert len(saved) == i + 1
    assert view.count == 8
    assert closed == [True]
