from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.execute_reach import MotionStopped, execute_target, motor_command
from deskbuddy.kinematics import JOINTS, Arm


class Clock:
    def __init__(self):
        self.time = 0.0

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class Bus:
    is_calibrated = True

    def __init__(
        self,
        config,
        *,
        enabled=True,
        bad_tracking=False,
        static_elbow_offset=0.0,
        tracking_error_deg=12.0,
    ):
        self.config = config
        self.goals = motor_command(np.deg2rad(config["preview_start_deg"]), config)
        self.writes = []
        self.enabled = enabled
        self.bad_tracking = bad_tracking
        self.tracking_error_deg = tracking_error_deg
        self.position_reads = 0
        self.static_elbow_offset = static_elbow_offset

    def sync_read(self, register, motors=None, **kwargs):
        if register == "Torque_Enable":
            return {j: int(self.enabled) for j in JOINTS}
        if register == "Operating_Mode":
            return {j: 0 for j in JOINTS}
        if register == "Present_Temperature":
            return {j: 25 for j in JOINTS}
        if register == "Goal_Position":
            return self.goals.copy()
        if register == "Present_Position":
            self.position_reads += 1
            result = self.goals.copy()
            result["elbow_flex"] += self.static_elbow_offset
            if self.bad_tracking and self.position_reads >= 5:
                result["shoulder_lift"] += self.tracking_error_deg
            return result
        raise AssertionError(register)

    def sync_write(self, register, values, **kwargs):
        assert register == "Goal_Position"
        assert set(values) == set(JOINTS)
        self.writes.append(values.copy())
        self.goals = values.copy()


def setup():
    config = deepcopy(DEFAULT)
    config.update(
        calibration_verified=True,
        joint_mapping_verified=True,
        joint_offsets_deg=[0, -3, -4, -5, 0],
    )
    start = np.deg2rad(config["preview_start_deg"])
    target = Arm().fk(start + np.deg2rad([2, 3, -2, 1, 0]))
    return config, target


def test_reach_inverts_software_offsets_and_never_enables_or_releases_torque():
    config, target = setup()
    bus = Bus(config)
    clock = Clock()
    plan, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert (
        result["completed"] and result["hold_retained"] and result["gripper_unchanged"]
    )
    assert result["execution_completed"] and result["model_target_reached"]
    assert result["settled_model_endpoint_error_mm"] < 2
    assert np.allclose(
        list(bus.writes[0].values()),
        list(motor_command(plan.joints_rad[0], config).values()),
    )
    assert np.all(plan.joints_rad[:, 4] == plan.joints_rad[0, 4])
    assert result["max_command_speed_deg_s"] == min(30, config["max_joint_speed_deg_s"])
    assert result["control_timing"]["planned_period_ms"] > 0
    assert "p99" in result["control_timing"]["service_ms"]


def test_stop_and_bad_tracking_abort_without_releasing_hold():
    config, target = setup()
    for tracking in (False, True):
        bus = Bus(config, bad_tracking=tracking)
        initial = bus.goals.copy()
        clock = Clock()

        def stop():
            return not tracking and len(bus.writes) >= 3

        with pytest.raises((MotionStopped, RuntimeError)):
            execute_target(
                bus, target, config, stop, clock=clock.now, sleep=clock.sleep
            )
        assert len(bus.writes) < 10
        assert bus.enabled
        # A stop must not replace the hold target with deflected feedback.
        assert max(abs(bus.goals[j] - initial[j]) for j in JOINTS) < 1


def test_missing_hold_blocks_all_motion_writes():
    config, target = setup()
    bus = Bus(config, enabled=False)
    with pytest.raises(ValueError, match="already be holding"):
        execute_target(bus, target, config, lambda: False)
    assert not bus.writes


def test_ui_disable_during_a_reach_stops_writes_and_retains_hold(tmp_path, monkeypatch):
    from deskbuddy.hardware_interlock import disable_motion, motion_disabled

    monkeypatch.chdir(tmp_path)
    config, target = setup()
    bus = Bus(config)
    clock = Clock()

    def stop():
        if len(bus.writes) >= 3:
            disable_motion()
        return motion_disabled()

    with pytest.raises(RuntimeError, match="Operator requested stop"):
        execute_target(bus, target, config, stop, clock=clock.now, sleep=clock.sleep)
    assert len(bus.writes) == 3
    assert bus.enabled
    assert motion_disabled()


def test_static_load_error_continues_smoothly_from_existing_command():
    config, target = setup()
    bus = Bus(config, static_elbow_offset=2.8)
    original = bus.goals.copy()
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["completed"] and result["started_from_existing_hold_target"]
    assert result["initial_tracking_error_deg"][2] == pytest.approx(2.8)
    assert np.allclose(list(bus.writes[0].values()), list(original.values()))


def test_static_sag_is_compensated_and_does_not_trigger_a_startup_gate():
    config, target = setup()
    bus = Bus(config, static_elbow_offset=4.1)
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["completed"]
    assert result["initial_load_bias_deg"][2] == pytest.approx(4.1)
    assert result["baseline_reestimated_during_move"] is False


def test_twelve_degree_hardware_ceiling_is_used_for_a_long_reach():
    config, _ = setup()
    config["max_joint_speed_deg_s"] = 12.0
    start = np.deg2rad(config["preview_start_deg"])
    target = Arm().fk(start + np.deg2rad([-50, 0, 0, 0, 0]))
    bus = Bus(config)
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert 11.6 < result["max_planned_command_speed_deg_s"] <= 12.0
    assert result["control_timing"]["max_sent_speed_deg_s"] <= 12.0 + 1e-8
    assert result["running_tracking_limits_deg"][2] == 10
    assert result["model_target_reached"]
    bus = Bus(config, static_elbow_offset=6.2)
    initial = bus.goals.copy()
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["completed"] and result["model_target_reached"]
    assert np.allclose(list(bus.writes[0].values()), list(initial.values()))
    assert result["initial_tracking_error_deg"][2] == pytest.approx(6.2)


def test_new_tracking_loss_still_stops_a_move_with_large_static_offset():
    config, target = setup()
    bus = Bus(config, static_elbow_offset=6.2, bad_tracking=True)
    clock = Clock()
    with pytest.raises(RuntimeError, match="tracking error"):
        execute_target(
            bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
        )
    assert len(bus.writes) < 5
    assert bus.enabled


def test_small_steps_do_not_hide_accumulated_error_when_a_joint_stalls():
    config, _ = setup()
    start = np.deg2rad(config["preview_start_deg"])
    target = Arm().fk(start + np.deg2rad([20, 0, 0, 0, 0]))

    class StalledBus(Bus):
        def sync_read(self, register, motors=None, **kwargs):
            result = super().sync_read(register, motors, **kwargs)
            if register == "Present_Position":
                result["shoulder_pan"] = self.fixed_pan
            return result

    bus = StalledBus(config, static_elbow_offset=6.2)
    bus.fixed_pan = bus.goals["shoulder_pan"]
    clock = Clock()
    with pytest.raises(RuntimeError, match="shoulder_pan tracking error"):
        execute_target(
            bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
        )
    assert len(bus.writes) > 10
    steps = np.diff([row["shoulder_pan"] for row in bus.writes])
    assert np.max(np.abs(steps)) < 0.4
    assert bus.enabled


def test_valid_workspace_target_is_not_blocked_by_old_trial_caps():
    config, _ = setup()
    start = np.deg2rad(config["preview_start_deg"])
    target = Arm().fk(start + np.deg2rad([-50, 0, 0, 0, 0]))
    assert np.linalg.norm(target - Arm().fk(start)) > 0.15
    bus = Bus(config)
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["completed"]


def test_execution_and_endpoint_report_use_the_calibrated_jaw_tip():
    config, _ = setup()
    config["tool_offset_m"] = [0.013, 0.003, 0]
    start = np.deg2rad(config["preview_start_deg"])
    arm = Arm(tool_offset_m=config["tool_offset_m"])
    target = arm.fk(start + np.deg2rad([2, 3, -2, 1, 0]))
    bus = Bus(config, static_elbow_offset=2.8)
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["model_target_reached"]
    assert result["settled_model_endpoint_error_mm"] < 2


@pytest.mark.parametrize("proposal_method", ["imf", "mlp"])
def test_flow_assisted_reach_uses_checked_refinement_and_existing_hold(
    monkeypatch, proposal_method
):
    pytest.importorskip("torch")
    from deskbuddy.learned import inference

    config, target = setup()
    config["planner_mode"] = "flow_assisted"
    config["proposal_method"] = proposal_method
    start = np.deg2rad(config["preview_start_deg"])
    expected = Arm().ik(target, start, locked_joints=["wrist_roll"])

    class Proposal:
        def predict(self, *args):
            return expected + np.deg2rad([0.2, 0.1, -0.1, 0.1, 0])

    monkeypatch.setattr(inference, "get_proposal", lambda *args, **kwargs: Proposal())
    bus, clock = Bus(config), Clock()
    initial = bus.goals.copy()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["planner_info"][-1]["used"] == "flow_assisted"
    assert result["planner_info"][-1]["proposal_method"] == proposal_method
    assert result["model_target_reached"] and result["hold_retained"]
    assert np.allclose(list(bus.writes[0].values()), list(initial.values()))


@pytest.mark.parametrize("error_deg", [4.04, 9.5])
def test_requested_tracking_tolerance_allows_error_below_ten_degrees(error_deg):
    config, target = setup()
    bus = Bus(config, bad_tracking=True, tracking_error_deg=error_deg)
    clock = Clock()
    _, result = execute_target(
        bus, target, config, lambda: False, clock=clock.now, sleep=clock.sleep
    )
    assert result["execution_completed"]
    assert result["running_tracking_limits_deg"] == [10] * 5
    assert result["substeps_per_original_step"] == 2
    assert result["max_planned_command_speed_deg_s"] <= 12.0
    assert result["max_planned_command_speed_deg_s"] > 1.5


def test_measured_start_offset_is_blended_without_a_velocity_spike():
    from copy import deepcopy

    import numpy as np

    from deskbuddy.config import DEFAULT
    from deskbuddy.execute_reach import start_from_measured
    from deskbuddy.kinematics import Arm
    from deskbuddy.planning import Plan, interpolate_joints, validate_plan

    arm = Arm()
    config = deepcopy(DEFAULT)
    config.update(control_hz=250, max_joint_speed_deg_s=24)
    start = np.deg2rad(config["preview_start_deg"])
    goal = start + np.deg2rad([10, 5, -5, 0, 0])
    times, q = interpolate_joints(start, goal, config)
    plan = Plan(times, q, arm.fk(goal), 0.0)
    measured = start + np.deg2rad([0.4, -0.3, 0.2, 0.1, 0.0])
    naive = deepcopy(plan)
    naive.joints_rad[0] = measured
    with pytest.raises(ValueError, match="velocity"):
        validate_plan(arm, naive, config)
    anchored = start_from_measured(plan, measured, config)
    assert np.allclose(anchored.joints_rad[0], measured)
    assert np.allclose(anchored.joints_rad[-1], goal)
    validate_plan(arm, anchored, config)
