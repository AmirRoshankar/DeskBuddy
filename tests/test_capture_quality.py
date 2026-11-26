import numpy as np
import pytest

from deskbuddy.capture_quality import calibration_depth


def test_stable_depth_passes_in_metres():
    frames = [
        np.full((480, 640), 0.5 + offset) for offset in (0, 0.001, -0.001, 0.002, 0)
    ]
    value, stats = calibration_depth(frames, 320, 240)
    assert value == pytest.approx(0.5)
    assert stats["depth_spread_mm"] == pytest.approx(3.0)


@pytest.mark.parametrize("depth", [0.0, 0.162, 0.223, 1.2, float("nan")])
def test_missing_or_close_range_rejected(depth):
    frames = [np.full((480, 640), depth)] * 5
    with pytest.raises(ValueError, match="30–100"):
        calibration_depth(frames, 320, 240)


def test_border_background_mismatch_and_temporal_noise_rejected():
    frames = [np.full((480, 640), 0.5)] * 5
    with pytest.raises(ValueError, match="edges"):
        calibration_depth(frames, 320, 32)
    frames[0][240, 320] = 0.52
    with pytest.raises(ValueError, match="disagree"):
        calibration_depth(frames, 320, 240)
    frames = [np.full((480, 640), v) for v in (0.5, 0.5, 0.5, 0.51, 0.5)]
    with pytest.raises(ValueError, match="fluctuating"):
        calibration_depth(frames, 320, 240)
