from copy import deepcopy

import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.hold_position import hold_supported_pose
from deskbuddy.kinematics import JOINTS


class Bus:
    is_calibrated = True

    def __init__(self, *, bad_readback=False, drift=False):
        self.log = []
        self.bad_readback = bad_readback
        self.drift = drift
        self.reads = 0
        self.enabled = False
        self.raw = {j: 2047 for j in JOINTS}
        self.goals = {}

    def sync_read(self, register, motors=None, *, normalize=True, num_retry=0):
        self.log.append(("read", register))
        names = list(JOINTS) if motors is None else motors
        if register == "Operating_Mode":
            return {j: 0 for j in names}
        if register == "Torque_Enable":
            return {j: int(self.enabled) for j in names}
        if register == "Goal_Position":
            return {j: v + int(self.bad_readback) for j, v in self.goals.items()}
        if register == "Present_Position":
            self.reads += 1
            if normalize:
                return {j: 0 for j in names}
            return {
                j: v + (20 if self.drift and self.reads > 1 else 0)
                for j, v in self.raw.items()
            }
        raise AssertionError(register)

    def write(self, register, name, value, **kwargs):
        self.log.append(("write", register, name, value))
        assert register == "Goal_Position"
        self.goals[name] = value

    def enable_torque(self, names, **kwargs):
        self.log.append(("enable", tuple(names)))
        assert self.goals == self.raw
        self.enabled = True


def test_exact_measured_goals_are_verified_before_enabling_arm_only():
    bus = Bus()
    report = hold_supported_pose(bus, deepcopy(DEFAULT), sleep=lambda _: None)
    assert report["holding"] and not report["reach_executed"]
    assert report["held_raw_positions"] == bus.raw
    enable = next(i for i, c in enumerate(bus.log) if c[0] == "enable")
    assert ("read", "Goal_Position") in bus.log[:enable]
    assert "gripper" not in bus.log[enable][1]
    assert not any(c[0] == "write" for c in bus.log[enable + 1 :])


@pytest.mark.parametrize("kwargs", [{"bad_readback": True}, {"drift": True}])
def test_bad_readback_or_moving_pose_never_enables_torque(kwargs):
    bus = Bus(**kwargs)
    with pytest.raises(ValueError):
        hold_supported_pose(bus, deepcopy(DEFAULT), sleep=lambda _: None)
    assert not bus.enabled


def test_supported_hold_does_not_depend_on_planner_joint_ranges():
    bus = Bus()
    config = deepcopy(DEFAULT)
    config["model_joint_limits_deg"] = [[10] * 5, [20] * 5]
    assert hold_supported_pose(bus, config, sleep=lambda _: None)["holding"]


def test_disable_during_hold_preparation_never_enables_torque():
    bus = Bus()

    def stop():
        return len(bus.goals) >= 3

    with pytest.raises(RuntimeError, match="Hold enable stopped"):
        hold_supported_pose(
            bus, deepcopy(DEFAULT), sleep=lambda _: None, stop_requested=stop
        )
    assert not bus.enabled and len(bus.goals) == 3


def test_disable_after_hold_enable_retains_hold():
    bus = Bus()
    with pytest.raises(RuntimeError, match="Hold enable stopped"):
        hold_supported_pose(
            bus,
            deepcopy(DEFAULT),
            sleep=lambda _: None,
            stop_requested=lambda: bus.enabled,
        )
    assert bus.enabled


@pytest.mark.parametrize("invalid", [float("nan"), 4096, -1, 2047.5])
def test_invalid_raw_feedback_never_writes_or_enables(invalid):
    bus = Bus()
    bus.raw["shoulder_lift"] = invalid
    with pytest.raises(ValueError, match="Invalid raw encoder"):
        hold_supported_pose(bus, deepcopy(DEFAULT), sleep=lambda _: None)
    assert not bus.enabled and not bus.goals
