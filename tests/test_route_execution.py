import time

import numpy as np
import pytest
from test_execute_reach import Bus, Clock, setup

from deskbuddy import execute_reach
from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.execution import (
    check_swept_obstacles,
    lift_route,
    require_same_scene,
    route_info,
)


def route_fixture():
    config, target = setup()
    arm = Arm()
    start = np.deg2rad(config["preview_start_deg"])
    initial = arm.fk(start)
    mid = (initial[:2] + target[:2]) / 2 + [0, 0.003]
    path = np.array([initial[:2], mid, target[:2]])
    plan = lift_route(
        arm,
        start,
        target,
        path,
        config,
        method="imf",
        candidates=4,
        boxes=[],
        scene_revision=7,
    )
    return config, arm, start, target, plan


def test_lift_preserves_route_and_times_joints_with_speed_limit():
    config, arm, start, target, plan = route_fixture()
    assert np.allclose(plan.joints_rad[0], start)
    assert (
        np.linalg.norm(arm.fk(plan.joints_rad[-1]) - target) < config["ik_tolerance_m"]
    )
    speed = np.abs(np.diff(plan.joints_rad, axis=0) / np.diff(plan.time_s)[:, None])
    assert speed.max() <= np.deg2rad(config["max_joint_speed_deg_s"]) * 1.001
    assert route_info(plan)["trajectory_method"] == "imf"
    assert not route_info(plan)["fallback"]


def test_worker_executes_selected_route_without_endpoint_replanning(monkeypatch):
    config, arm, start, target, plan = route_fixture()

    def forbidden(*a, **kw):
        raise AssertionError("Must not replace selected route with direct IK")

    monkeypatch.setattr(execute_reach, "compensated_plan", forbidden)
    bus, clock = Bus(config), Clock()
    executed, result = execute_reach.execute_target(
        bus,
        target,
        config,
        lambda: False,
        clock=clock.now,
        sleep=clock.sleep,
        selected_plan=plan,
    )
    assert result["completed"] and bus.writes
    assert np.allclose(executed.joints_rad, plan.joints_rad)


def test_stale_start_blocks_selected_route_before_motor_write():
    config, arm, start, target, plan = route_fixture()
    plan.joints_rad[0, 0] += np.deg2rad(3)
    bus, clock = Bus(config), Clock()
    with pytest.raises(ValueError, match="Arm moved"):
        execute_reach.execute_target(
            bus,
            target,
            config,
            lambda: False,
            clock=clock.now,
            sleep=clock.sleep,
            selected_plan=plan,
        )
    assert not bus.writes


def test_collision_between_joint_samples_is_rejected():
    class FakeArm:
        def frames(self, q):
            y = q[0]
            return np.array([[0.0, y, 0.1], [0.2, y, 0.1]]), None

    obstacle = dict(low_m=[0.09, -0.01, 0.07], high_m=[0.11, 0.01, 0.13])
    with pytest.raises(ValueError, match="overlaps"):
        check_swept_obstacles(
            FakeArm(), [np.array([-0.1]), np.array([0.1])], [obstacle]
        )


def test_scene_revision_alone_does_not_block_but_monitor_dropout_does():
    route = dict(scene_revision=7, obstacles_m=[])
    live = dict(
        unix_time_s=time.time(),
        obstacle_detection=dict(scene_revision=7),
        obstacle_boxes=[],
    )
    require_same_scene(route, live)
    live["obstacle_detection"]["scene_revision"] = 8
    require_same_scene(route, live)
    assert route["scene_revision"] == 8
    live["unix_time_s"] = 0
    with pytest.raises(RuntimeError, match="monitoring unavailable"):
        require_same_scene(route, live)


def test_worker_accepts_changed_object_away_from_the_route_without_motor_enable():
    config, _, _, target, plan = route_fixture()
    bus = Bus(config)
    clock = Clock()
    actual = dict(low_m=[0.8, 0.8, 0.1], high_m=[0.9, 0.9, 0.2])
    live = dict(
        unix_time_s=time.time(),
        obstacle_detection=dict(scene_revision=999),
        obstacle_boxes=[actual],
    )
    _, result = execute_reach.execute_target(
        bus,
        target,
        config,
        lambda: False,
        clock=clock.now,
        sleep=clock.sleep,
        selected_plan=plan,
        read_scene=lambda: live,
    )
    assert result["completed"] and result["hold_retained"]
    assert result["planner_info"][0]["obstacles_m"] == [actual]
