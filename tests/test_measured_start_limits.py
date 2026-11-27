import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.execute_reach import start_from_measured
from deskbuddy.kinematics import Arm
from deskbuddy.planning import Plan


def test_encoder_blend_cannot_push_near_limit_path_outward():
    arm = Arm()
    start = np.zeros(5)
    start[0] = arm.upper[0] - np.deg2rad(0.3)
    goal = start.copy()
    goal[0] = arm.upper[0]
    times = np.linspace(0, 1, 251)
    joints = start + (goal - start) * np.clip(times[:, None] / 0.2, 0, 1)
    measured = start.copy()
    measured[0] += np.deg2rad(0.2)
    plan = Plan(times, joints, arm.fk(goal), 0.0)
    corrected = start_from_measured(plan, measured, DEFAULT)
    assert np.all(corrected.joints_rad <= arm.upper + 1e-12)
    assert np.allclose(corrected.joints_rad[0], measured)
    assert np.allclose(corrected.joints_rad[-1], goal)
    assert (
        np.max(
            np.abs(np.diff(corrected.joints_rad, axis=0))
            / np.diff(corrected.time_s)[:, None]
        )
        <= np.deg2rad(DEFAULT["max_joint_speed_deg_s"]) * 1.001
    )


def test_outside_measured_start_preserved_and_recovers_monotonically():
    arm = Arm()
    start = np.zeros(5)
    start[0] = arm.upper[0] - np.deg2rad(0.1)
    goal = start.copy()
    goal[0] -= np.deg2rad(5)
    times = np.linspace(0, 1, 251)
    joints = start + (goal - start) * times[:, None]
    measured = start.copy()
    measured[0] = arm.upper[0] + np.deg2rad(0.2)
    plan = start_from_measured(
        Plan(times, joints, arm.fk(goal), 0.0), measured, DEFAULT
    )
    excess = np.maximum(plan.joints_rad - arm.upper, 0)
    assert np.all(np.diff(excess, axis=0) <= 1e-12)
    assert np.allclose(plan.joints_rad[0], measured)
    assert excess[-1].max() == 0
