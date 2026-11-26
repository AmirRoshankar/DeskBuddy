import time
from types import SimpleNamespace

from deskbuddy import hold_control
from deskbuddy.hardware_interlock import disable_motion, motion_disabled
from deskbuddy.kinematics import JOINTS


class Value:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


class Bar:
    def step(self, *args):
        self.args = args

    def grid_remove(self):
        self.hidden = True


def fixture():
    state = dict(
        motor=dict(torque_enabled={j: 0 for j in JOINTS}),
        last_state_read=time.monotonic(),
        calibration_job=None,
        plan=object(),
        robot_io_paused=False,
    )
    root = SimpleNamespace(after=lambda *args: None)
    status, bar = Value(), Bar()
    return hold_control.HoldControl(root, {}, state, status, bar), state, status, bar


def test_cancelled_ui_confirmation_preserves_disable_and_never_launches(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    disable_motion()
    flow, state, _, _ = fixture()
    monkeypatch.setattr(hold_control.messagebox, "askyesno", lambda *a, **k: False)
    monkeypatch.setattr(
        hold_control,
        "launch_calibration_action",
        lambda *a: (_ for _ in ()).throw(AssertionError("No worker")),
    )
    flow.start()
    assert motion_disabled() and state["calibration_job"] is None


def test_confirmed_enable_starts_hold_only_and_requires_clearance(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    disable_motion()
    flow, state, status, bar = fixture()
    calls = []
    job = SimpleNamespace()
    monkeypatch.setattr(hold_control.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(
        hold_control,
        "launch_calibration_action",
        lambda action, config: calls.append(action) or job,
    )
    flow.start()
    assert calls == ["hold"] and not motion_disabled()
    assert state["hold_pending_clear"] and state["plan"] is None
    flow.finish()
    assert state["hold_pending_clear"]  # No confirmed active hold yet.
    state["motor"]["torque_enabled"] = {j: 1 for j in JOINTS}
    state["calibration_job"] = None
    state["last_state_read"] = time.monotonic()
    flow.finish()
    assert not state["hold_pending_clear"] and bar.hidden
    assert "Enabled" in status.value


def test_disable_releases_only_after_motion_worker_finished_and_confirms_all_motors(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    flow, state, status, bar = fixture()
    state["motion_job"] = object()
    calls = []
    job = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 0, returncode=0),
        finish=lambda: dict(
            released=True, torque_enabled={j: 0 for j in [*JOINTS, "gripper"]}
        ),
    )
    monkeypatch.setattr(
        hold_control,
        "launch_calibration_action",
        lambda action, config: calls.append(action) or job,
    )
    flow.release()
    flow.release_when_idle()
    assert motion_disabled() and not calls
    state["motion_job"] = None
    flow.release_when_idle()
    flow.poll_release()
    assert calls == ["release"] and not state["release_pending"]
    assert "six motors released" in status.value
