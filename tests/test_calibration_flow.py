import time
from types import SimpleNamespace

import numpy as np
import pytest

from deskbuddy import calibration_flow as module
from deskbuddy.calibration_action import supported_action
from deskbuddy.calibration_session import checked_fit
from deskbuddy.config import load_config
from deskbuddy.kinematics import JOINTS


class Bar:
    def __init__(self, root):
        self.message = ""

    def step(self, message, *args, **kwargs):
        self.message = message

    def grid_remove(self):
        pass


def test_whole_workflow_requires_support_fit_paper_hold_and_final_confirmation(
    tmp_path, monkeypatch
):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "CalibrationBar", Bar)
    config = load_config()
    config["camera_serial"] = "cam"
    transform = np.diag([1.0, -1, -1, 1])
    transform[:3, 3] = [0.25, 0, 0.7]
    camera = np.random.default_rng(4).uniform(
        [-0.1, -0.1, 0.35], [0.1, 0.1, 0.55], (8, 3)
    )
    rows = [
        dict(
            session_id="new",
            camera_serial="cam",
            camera_point_m=p.tolist(),
            base_point_model_m=(transform[:3, :3] @ p + transform[:3, 3]).tolist(),
        )
        for p in camera
    ]
    intr = dict(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    state = dict(
        tool_points=rows,
        motion_job=None,
        motor=None,
        last_state_read=0.0,
        frame=(np.zeros((480, 640, 3)), np.full((480, 640), 0.6), intr),
    )
    status = SimpleNamespace(set=lambda _: None)
    root = SimpleNamespace(after=lambda *args: None)
    operations, finished = [], []

    class Job:
        process = SimpleNamespace(poll=lambda: 0, returncode=0)

        def finish(self):
            return {"ok": True}

    monkeypatch.setattr(
        module,
        "launch_calibration_action",
        lambda action, config: operations.append(action) or Job(),
    )
    flow = module.CalibrationFlow(
        root,
        config,
        state,
        status,
        "config.json",
        lambda: None,
        lambda: finished.append(True),
        lambda: None,
    )
    flow.begin()
    assert not operations and not config["calibration_verified"]
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *args, **kwargs: False)
    flow.support_action("release")
    assert not operations
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *args, **kwargs: True)
    flow.support_action("release")
    flow.poll("release")
    assert flow.stage == "capture" and operations == ["release"]
    flow.assessment("Fit passed")
    flow.use_fit()
    assert flow.stage == "paper" and not config["calibration_verified"]
    obj = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])
    k = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    pixels = cv2.projectPoints(
        obj, np.array([0.0, 0, 0.2]), np.array([-0.105, -0.1485, 0.6]), k, None
    )[0].reshape(-1, 2)
    for pixel in pixels:
        flow.capture_paper(pixel)
    flow.use_paper()
    assert flow.stage == "hold" and not config["calibration_verified"]
    flow.support_action("hold")
    flow.poll("hold")
    assert operations == ["release", "hold"] and not config["calibration_verified"]
    flow.finish()
    assert not finished  # No valid live hold feedback yet.
    state.update(
        motor=dict(torque_enabled=dict.fromkeys(JOINTS, 1)),
        last_state_read=time.monotonic(),
    )
    flow.finish()
    assert finished and config["calibration_verified"] and flow.stage is None


def test_wrong_camera_fit_and_uncalibrated_motor_release_are_blocked():
    config = load_config()
    with pytest.raises(ValueError, match="eight observations"):
        checked_fit([], config)

    class Bus:
        is_calibrated = False

        def disable_torque(self, **kwargs):
            pytest.fail("Must not touch motors with mismatched calibration")

    with pytest.raises(ValueError, match="calibration"):
        supported_action(Bus(), config, "release")


def test_release_verifies_all_six_motors_and_never_writes_position_goals():
    class Bus:
        is_calibrated = True
        released = False

        def disable_torque(self, **kwargs):
            self.released = True

        def sync_read(self, register, **kwargs):
            assert register == "Torque_Enable"
            return dict.fromkeys([*JOINTS, "gripper"], 0)

    bus = Bus()
    assert supported_action(bus, load_config(), "release")["released"]
    assert bus.released
