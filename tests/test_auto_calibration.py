import time

import numpy as np
import pytest

from deskbuddy.config import load_config
from deskbuddy.kinematics import Arm
from deskbuddy.ui.auto_calibration import AutoCalibration


class Status:
    def set(self, message):
        self.message = message


def test_auto_calibration_requires_initial_pose_before_any_motion(monkeypatch):
    from deskbuddy import multiview

    monkeypatch.setattr(multiview, "calibration_current", lambda config: True)
    config = load_config()
    config.update(
        calibration_verified=True,
        secondary_camera_calibration={"metric_3d_validated": False},
    )

    def forbidden(*args):
        raise AssertionError("Motion must not start")

    operation = AutoCalibration(None, config, {}, Arm(), Status(), forbidden, None)
    with pytest.raises(ValueError, match="initial camera alignment"):
        operation.start()


def test_missing_marker_stops_without_motion():
    state = dict(
        frame=(np.zeros((60, 60, 3), np.uint8), None, None),
        secondary_frame=(np.zeros((60, 60, 3), np.uint8), time.time()),
        last_state_read=time.monotonic(),
        q=np.zeros(5),
    )
    status = Status()
    operation = AutoCalibration(
        None, {}, state, Arm(), status, lambda *args: pytest.fail("Must not move"), None
    )
    operation.capture()
    assert "Webcam tip identity" in status.message
    assert operation.samples == []


def test_unverified_camera_estimate_can_start_rgb_collection(monkeypatch):
    from deskbuddy.kinematics import JOINTS
    from deskbuddy.ui import auto_calibration

    config = load_config()
    config.update(
        calibration_verified=False,
        T_base_camera=np.eye(4).tolist(),
        secondary_camera_calibration=dict(
            T_base_camera=np.eye(4).tolist(),
            intrinsics=dict(fx=600, fy=600, ppx=320, ppy=240),
            metric_3d_validated=False,
        ),
    )
    state = dict(
        q=np.zeros(5),
        motor=dict(torque_enabled={name: 1 for name in JOINTS}),
        frame=(None, None, None),
        secondary_frame=(None, time.time()),
        tip_tracking=dict(valid=True),
    )
    monkeypatch.setattr(auto_calibration.messagebox, "askyesno", lambda *args: True)
    operation = AutoCalibration(
        None,
        config,
        state,
        Arm(),
        Status(),
        lambda *args: pytest.fail("No hardware"),
        None,
    )
    captured = []
    operation.capture = lambda: captured.append(True)
    operation.begin_preflight = lambda start: operation.capture()
    operation.start()
    assert captured == [True]
    assert operation.bootstrap
    assert not config["calibration_verified"]
    assert not config["secondary_camera_calibration"]["metric_3d_validated"]


def test_webcam_tip_suffices_when_realsense_has_no_orange():
    from types import SimpleNamespace

    primary = np.zeros((100, 100, 3), np.uint8)
    webcam = primary.copy()
    webcam[40:60, 40:60] = [255, 100, 0]
    intr = dict(fx=100, fy=100, ppx=50, ppy=50)
    state = dict(
        frame=(primary, None, intr),
        secondary_frame=(webcam, time.time()),
        last_state_read=time.monotonic(),
        q=np.zeros(5),
    )
    status = Status()
    executor = SimpleNamespace(submit=lambda *args: None)
    root = SimpleNamespace(after=lambda *args: None)
    operation = AutoCalibration(
        root,
        dict(T_base_camera=np.eye(4).tolist()),
        state,
        Arm(),
        status,
        lambda *args: pytest.fail("No physical launch"),
        executor,
    )
    operation.targets = [np.zeros(5), np.zeros(5)]
    operation.rgb_observations = []
    operation.capture()
    assert operation.index == 1
    assert operation.rgb_observations[0][0] == [None, None]
    assert operation.rgb_observations[0][1] == [49.5, 49.5]
    assert "continuing" in status.message
