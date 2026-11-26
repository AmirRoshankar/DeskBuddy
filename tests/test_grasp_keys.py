import numpy as np
import pytest
from test_paper_surface import calibrated_paper

from deskbuddy.grasp_keys import detect_key_target, run_sequence, vision_grasped
from deskbuddy.overlays import project


def test_grasp_order_and_success_after_one_retry():
    events = []
    attempts = [False, True]
    result = run_sequence(
        lambda target, contact: events.append(("move", target.copy(), contact)),
        lambda p: events.append(("gripper", p)),
        lambda point: attempts.pop(0),
        np.array([0.2, 0.1, 0]),
        np.array([0.2, 0.1, 0.1]),
        0.005,
        lambda: False,
    )
    assert result == dict(grasp_confirmed=True, attempts=2, retries=1)
    assert events[0] == ("gripper", 70.0)
    assert events[2][0] == "move" and events[2][2]
    assert events[3] == ("gripper", 0.0)
    assert events[4][1][2] == 0.1


def test_five_retries_are_bounded_and_stop_prevents_commands():
    grips = []
    result = run_sequence(
        lambda *a: None,
        grips.append,
        lambda p: False,
        [0.2, 0.1, 0],
        [0.2, 0.1, 0.1],
        0.005,
        lambda: False,
    )
    assert (
        result["attempts"] == 6
        and result["retries"] == 5
        and not result["grasp_confirmed"]
    )
    assert len(grips) == 12 and grips[-1] == 0
    with pytest.raises(RuntimeError, match="stopped"):
        run_sequence(
            lambda *a: pytest.fail("motion after stop"),
            lambda p: pytest.fail("gripper after stop"),
            lambda p: False,
            [0.2, 0.1, 0],
            [0.2, 0.1, 0.1],
            0.005,
            lambda: True,
        )


def test_table_keys_target_uses_paper_plane_even_without_depth():
    c, i, _, _ = calibrated_paper()
    report = dict(
        key_detections=[
            dict(confidence=0.8, pixel_roi=[90, 90, 110, 110], position_m=None)
        ]
    )
    point, _ = detect_key_target(report, i, c)
    assert abs(point[2]) < 0.01
    with pytest.raises(ValueError, match="No confident keys"):
        detect_key_target({}, i, c)


def test_grasp_verification_requires_positive_key_evidence_not_disappearance():
    c, i, _, _ = calibrated_paper()
    source = np.array([0.1, 0.1, 0])
    hover = source + [0, 0, 0.1]
    ref = np.array(project(source, c["T_base_camera"], i))
    tip = np.array(project(hover, c["T_base_camera"], i))
    live = dict(intrinsics=i, obstacle_detection=dict(key_detections=[]))
    assert not vision_grasped(live, source, hover, c, ref)

    live["obstacle_detection"]["key_detections"] = [
        dict(
            confidence=0.9,
            pixel_roi=[*(tip - 3), *(tip + 3)],
            position_m=hover.tolist(),
        )
    ]
    # This camera sees a nearly vertical lift with little image displacement:
    # insufficient evidence stays unconfirmed, even with one depth observation.
    if np.linalg.norm(tip - ref) > 12:
        assert vision_grasped(live, source, hover, c, ref)
    live["obstacle_detection"]["key_detections"][0]["pixel_roi"] = [
        *(ref - 3),
        *(ref + 3),
    ]
    assert not vision_grasped(live, source, hover, c, ref)


@pytest.mark.parametrize("stop_during_open", [False, True])
def test_gripper_adapter_enables_only_gripper_and_honors_stop(
    monkeypatch, stop_during_open, tmp_path
):
    import copy
    import time

    from test_execute_reach import Bus
    from test_route_execution import route_fixture

    from deskbuddy import execute_reach, grasp_keys
    from deskbuddy.demo import T_BASE_CAMERA

    config, arm, start, target, selected = route_fixture()
    config.update(
        T_base_camera=T_BASE_CAMERA.tolist(),
        grasp_request=dict(source_m=[*target[:2], 0]),
    )
    import json

    from deskbuddy.config import load_config

    serialized = tmp_path / "grasp-job.json"
    serialized.write_text(json.dumps(config))
    config = load_config(serialized)

    class GraspBus(Bus):
        def __init__(self, c):
            super().__init__(c)
            self.grip = 2.0
            self.grip_on = False
            self.grip_writes = []
            self.enabled_names = []

        def sync_read(self, register, motors=None, **kwargs):
            if motors == ["gripper"]:
                return {
                    "gripper": {
                        "Present_Position": self.grip,
                        "Goal_Position": self.grip,
                        "Operating_Mode": 0,
                        "Torque_Enable": int(self.grip_on),
                        "Present_Temperature": 25,
                    }[register]
                }
            return super().sync_read(register, motors, **kwargs)

        def sync_write(self, register, values, **kwargs):
            if set(values) == {"gripper"}:
                self.grip = values["gripper"]
                self.grip_writes.append(self.grip)
                return
            super().sync_write(register, values, **kwargs)

        def enable_torque(self, names, **kwargs):
            self.enabled_names.extend(names)
            self.grip_on = True

    bus = GraspBus(config)
    moves = []

    def phase(*args, **kwargs):
        from deskbuddy.grasp_geometry import oriented_ik

        plan = copy.deepcopy(selected)
        plan.target_m = np.array(args[2])
        goal = kwargs.get("goal_pose")
        if goal is None:
            goal = oriented_ik(args[0], args[2], args[1], [0, 0, -1], thorough=True)
        plan.joints_rad[-1] = goal
        return plan

    def fake_execute(bus, target, cfg, *args, **kwargs):
        moves.append(np.array(target))
        bus.goals = execute_reach.motor_command(
            kwargs["selected_plan"].joints_rad[-1], cfg
        )
        return kwargs["selected_plan"], dict(
            completed=True, settled_model_endpoint_error_mm=0
        )

    monkeypatch.setattr("deskbuddy.trajectory.execution.lift_route", phase)
    monkeypatch.setattr("deskbuddy.trajectory.joint_planner.plan_joint_route", phase)
    monkeypatch.setattr(execute_reach, "execute_target", fake_execute)
    monkeypatch.setattr(grasp_keys, "vision_grasped", lambda *args: True)

    def live():
        return dict(
            intrinsics=dict(fx=600, fy=600, ppx=320, ppy=240),
            obstacle_detection=dict(scene_revision=7, detection_unix_s=time.time()),
            obstacle_boxes=[],
        )

    def stopped():
        return stop_during_open and len(bus.grip_writes) > 2

    if stop_during_open:
        with pytest.raises(RuntimeError, match="stopped"):
            grasp_keys.execute_grasp(
                bus,
                selected,
                config,
                stopped,
                lambda _: None,
                live,
                sleep=lambda _: None,
            )
        assert not moves
    else:
        _, result = grasp_keys.execute_grasp(
            bus, selected, config, stopped, lambda _: None, live, sleep=lambda _: None
        )
        assert result["grasp_confirmed"] and result["attempts"] == 1 and moves
        assert max(bus.grip_writes) == 70 and bus.grip_writes[-1] == 0
    assert bus.enabled_names == ["gripper"] and bus.enabled


def test_grasp_pose_preflight_failure_does_not_enable_or_command_any_motor(monkeypatch):
    from test_execute_reach import Bus
    from test_route_execution import route_fixture

    from deskbuddy.grasp_keys import execute_grasp

    config, _, _, target, selected = route_fixture()
    config["grasp_request"] = dict(source_m=[*target[:2], 0])
    bus = Bus(config)

    def blocked(*args, **kwargs):
        raise ValueError("Alignment blocked")

    monkeypatch.setattr("deskbuddy.trajectory.joint_planner.plan_joint_route", blocked)
    with pytest.raises(ValueError, match="Alignment blocked"):
        execute_grasp(
            bus,
            selected,
            config,
            lambda: False,
            lambda _: None,
            lambda: dict(obstacle_boxes=[], obstacle_detection={}),
        )
    assert not bus.writes
