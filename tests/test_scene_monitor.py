import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from deskbuddy.trajectory.execution import require_same_scene, same_geometry
from deskbuddy.trajectory.scene_monitor import SceneMonitor, ScenePathBlocked


def box(x, y):
    return dict(low_m=[x, y, 0.08], high_m=[x + 0.03, y + 0.03, 0.22])


class Arm:
    def frames(self, q):
        return np.array([[q[0], 0.0, 0.1], [q[0], 0.0, 0.2]]), None


def monitor_fixture():
    live = dict(
        unix_time_s=time.time(),
        obstacle_boxes=[],
        obstacle_detection=dict(scene_revision=1),
    )
    route = dict(obstacles_m=[], scene_revision=1)
    plan = SimpleNamespace(
        time_s=np.array([0.0, 1.0, 2.0]), joints_rad=np.array([[0.0], [0.5], [1.0]])
    )
    tick = [0.0]
    notices = []
    monitor = SceneMonitor(
        Arm(), plan, route, lambda: live, notices.append, lambda: tick[0]
    )
    return monitor, live, route, tick, notices


def test_box_order_and_small_noise_are_not_scene_changes():
    boxes = [box(0.2, 1), box(0.6, 1)]
    changed = deepcopy(boxes[::-1])
    changed[0]["low_m"][0] += 0.004
    assert same_geometry(boxes, changed)
    route = dict(scene_revision=1, obstacles_m=boxes)
    require_same_scene(
        route,
        dict(
            unix_time_s=time.time(),
            obstacle_boxes=changed,
            obstacle_detection=dict(scene_revision=123),
        ),
    )
    assert route["scene_revision"] == 123


def test_new_object_clear_of_route_continues_and_updates_geometry():
    monitor, live, route, tick, notices = monitor_fixture()
    try:
        live["obstacle_boxes"] = [box(0.7, 1)]
        monitor.poll(np.array([0.0]), 0.0)
        monitor.future.result(timeout=3)
        tick[0] += 0.06
        monitor.poll(np.array([0.02]), 0.04)
        assert route["obstacles_m"] == live["obstacle_boxes"]
        assert notices == ["Scene updated · remaining path clear"]
    finally:
        monitor.close()


def test_object_blocking_remaining_route_requests_controlled_hold():
    monitor, live, _, tick, _ = monitor_fixture()
    try:
        live["obstacle_boxes"] = [box(0.7, -0.015)]
        monitor.poll(np.array([0.0]), 0.0)
        with pytest.raises(ValueError):
            monitor.future.result(timeout=3)
        tick[0] += 0.06
        with pytest.raises(ScenePathBlocked, match="blocks remaining path"):
            monitor.poll(np.array([0.02]), 0.04)
    finally:
        monitor.close()


def test_obsolete_collision_result_does_not_stop_for_object_that_moved_away():
    monitor, live, _, tick, _ = monitor_fixture()
    try:
        live["obstacle_boxes"] = [box(0.7, -0.015)]
        monitor.poll(np.array([0.0]), 0.0)
        with pytest.raises(ValueError):
            monitor.future.result(timeout=3)
        live["obstacle_boxes"] = []
        tick[0] += 0.06
        monitor.poll(np.array([0.02]), 0.04)
        assert monitor.future is None
    finally:
        monitor.close()


def test_change_behind_the_current_phase_is_not_part_of_remaining_path():
    monitor, live, _, tick, notices = monitor_fixture()
    try:
        live["obstacle_boxes"] = [box(0.1, -0.015)]
        monitor.poll(np.array([0.6]), 1.2)
        monitor.future.result(timeout=3)
        tick[0] += 0.06
        monitor.poll(np.array([0.62]), 1.24)
        assert notices
    finally:
        monitor.close()


def test_short_snapshot_read_error_retains_previous_scene():
    monitor, _, _, tick, _ = monitor_fixture()
    try:

        def missing():
            raise ValueError("partial JSON")

        monitor.read_scene = missing
        monitor.poll(np.array([0.0]), 0.0)
        tick[0] = 2.1
        with pytest.raises(RuntimeError, match="monitoring unavailable"):
            monitor.poll(np.array([0.0]), 0.0)
    finally:
        monitor.close()
