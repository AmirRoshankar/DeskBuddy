from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT, load_config
from deskbuddy.demo import synthetic_point
from deskbuddy.kinematics import Arm
from deskbuddy.planning import Plan, halve_steps, plan_reach, validate_plan
from deskbuddy.storage import load_plan, save_plan


@pytest.fixture
def setup():
    return Arm(), deepcopy(DEFAULT), np.deg2rad(DEFAULT["preview_start_deg"])


def test_fk_reference_and_limits(setup):
    arm, _, _ = setup
    # Reference from composing the pinned SO-101 URDF at q=0.
    assert np.allclose(
        arm.fk(np.zeros(5)), [0.391361470, -0.000009212, 0.226469710], atol=1e-8
    )
    assert len(arm.lower) == 5


def test_fk_ik_roundtrip_nearby_targets(setup):
    arm, config, start = setup
    rng = np.random.default_rng(3)
    for _ in range(12):
        known = start + np.deg2rad(rng.uniform(-10, 10, 5))
        target = arm.fk(known)
        q = arm.ik(target, start)
        assert np.linalg.norm(arm.fk(q) - target) < config["ik_tolerance_m"]
        assert (q >= arm.lower).all() and (q <= arm.upper).all()


def test_unreachable_ik_rejected(setup):
    arm, _, start = setup
    with pytest.raises(ValueError, match="IK failed"):
        arm.ik([3, 0, 0], start)


def test_baseline_and_serialization(setup, tmp_path):
    arm, config, start = setup
    target = synthetic_point(385, 195) + [0, 0, config["hover_m"]]
    p = plan_reach(arm, start, target, config)
    assert np.allclose(p.joints_rad[0], start)
    velocity = np.diff(p.joints_rad, axis=0) / np.diff(p.time_s)[:, None]
    assert np.rad2deg(np.abs(velocity)).max() <= config["max_joint_speed_deg_s"]
    path = tmp_path / "plan.json"
    save_plan(path, p, config, synthetic=True)
    loaded, data = load_plan(path)
    assert data["synthetic"] is True
    validate_plan(arm, loaded, config)
    assert np.allclose(loaded.joints_rad, p.joints_rad)


@pytest.mark.parametrize(
    "target", [[2, 0, 0.2], [0.2, 0, -0.1], [float("nan"), 0, 0.2]]
)
def test_bad_targets_rejected(setup, target):
    arm, config, start = setup
    with pytest.raises(ValueError):
        plan_reach(arm, start, target, config)


def test_speed_limit_and_table_screening(setup):
    arm, config, start = setup
    p = plan_reach(arm, start, synthetic_point(385, 195) + [0, 0, 0.06], config)
    p.time_s *= 0.01
    with pytest.raises(ValueError, match="velocity"):
        validate_plan(arm, p, config)
    config["table_z_m"] = 0.3
    with pytest.raises(ValueError, match="table"):
        plan_reach(arm, start, arm.fk(start), config)


def test_nan_joints_and_bad_timestamps_rejected(setup):
    arm, config, start = setup
    bad = np.tile(start, (2, 1))
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_plan(arm, Plan(np.array([0, 1]), bad, arm.fk(start), 0), config)
    with pytest.raises(ValueError, match="timestamps"):
        validate_plan(
            arm,
            Plan(np.array([0, 0]), np.tile(start, (2, 1)), arm.fk(start), 0),
            config,
        )


def test_config_rejects_zero_speed_and_unknown_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"max_joint_speed_deg_s":0}')
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text('{"calibrated":true}')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


def test_locked_wrist_roll_and_configured_measured_limits(setup):
    arm, config, start = setup
    target = arm.fk(start + np.deg2rad([3, 4, -3, 2, 0]))
    plan = plan_reach(arm, start, target, config)
    assert np.all(plan.joints_rad[:, 4] == start[4])
    limited = Arm([[-60, -80, -80, -70, -120], [60, 80, 80, 70, 120]])
    assert np.allclose(np.rad2deg(limited.lower), [-60, -80, -80, -70, -120])
    config["validated_wrist_roll_deg"] = 90
    with pytest.raises(ValueError, match="wrist roll"):
        plan_reach(arm, start, target, config)


def test_slightly_low_hold_gets_upward_entry_without_changing_target(setup, tmp_path):
    arm, config, seed = setup
    start = arm.ik([0.30, 0, 0.035], seed)
    target = np.array([0.29, 0.025, 0.075])
    p = plan_reach(arm, start, target, config)
    assert p.recovery_waypoint_m is not None
    assert p.recovery_waypoint_m[2] == pytest.approx(0.045)
    assert np.allclose(p.target_m, target)
    positions = np.array([arm.fk(q) for q in p.joints_rad])
    assert positions[:, 2].min() >= positions[0, 2] - 1e-5
    path = tmp_path / "recovery.json"
    save_plan(path, p, config, synthetic=True)
    loaded, _ = load_plan(path)
    assert np.allclose(loaded.recovery_waypoint_m, p.recovery_waypoint_m)


def test_recovery_cannot_worsen_the_violation_or_bypass_table_clearance(setup):
    arm, config, seed = setup
    low = arm.ik([0.30, 0, 0.035], seed)
    lower = arm.ik([0.30, 0, 0.03], low)
    end = arm.ik([0.30, 0, 0.075], low)
    with pytest.raises(ValueError, match="farther outside"):
        validate_plan(
            arm,
            Plan(
                np.array([0, 10, 20]),
                np.array([low, lower, end]),
                np.array([0.30, 0, 0.075]),
                0.0,
            ),
            config,
        )
    unsafe = arm.ik([0.30, 0, -0.005], seed)
    with pytest.raises(ValueError, match="table"):
        plan_reach(arm, unsafe, [0.30, 0, 0.075], config)
    with pytest.raises(ValueError, match=r"Selected target.*x=60.0"):
        plan_reach(arm, seed, [0.60, 0, 0.075], config)


def test_starting_outside_target_footprint_can_enter_without_widening_goals(setup):
    arm, config, seed = setup
    start = arm.ik([0.37, 0, 0.13], seed)
    config["workspace_max_m"][0] = 0.32
    target = np.array([0.30, 0.01, 0.12])
    plan = plan_reach(arm, start, target, config)
    assert plan.recovery_waypoint_m is not None
    positions = np.array([arm.fk(q) for q in plan.joints_rad])
    violation = np.maximum(positions[:, 0] - 0.32, 0)
    assert np.diff(violation).max() < 0.00005
    assert config["workspace_max_m"][0] == 0.32
    with pytest.raises(ValueError, match="Selected target"):
        plan_reach(arm, start, [0.37, 0, 0.13], config)


def test_subdivision_keeps_equal_half_steps_and_original_duration(setup):
    arm, config, start = setup
    target = arm.fk(start + np.deg2rad([3, 2, -2, 1, 0]))
    original = plan_reach(arm, start, target, config)
    refined = halve_steps(original)
    assert np.allclose(refined.joints_rad[::2], original.joints_rad)
    assert np.allclose(
        refined.joints_rad[1::2],
        (original.joints_rad[:-1] + original.joints_rad[1:]) / 2,
    )
    assert np.allclose(
        np.diff(refined.time_s), np.repeat(np.diff(original.time_s) / 2, 2)
    )
    assert refined.time_s[-1] == pytest.approx(original.time_s[-1])
    validate_plan(arm, refined, config)
