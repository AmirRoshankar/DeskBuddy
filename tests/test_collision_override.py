import numpy as np
import pytest
from test_clearance_regression import replay
from test_execute_reach import Bus, Clock

from deskbuddy.adaptive_control import HARDWARE_SPEED_LIMIT_DEG_S
from deskbuddy.execute_reach import execute_target
from deskbuddy.trajectory.execution import (
    RoutePlanner,
    require_same_scene,
    route_info,
)
from deskbuddy.ui.scene_inspector import OrbitView, render_scene


def override_plan():
    data, config, arm, boxes = replay()
    config["allow_object_collisions"] = True
    config["max_joint_speed_deg_s"] = HARDWARE_SPEED_LIMIT_DEG_S
    config["preview_start_deg"] = np.rad2deg(data["start"]).tolist()
    config["calibration_verified"] = config["joint_mapping_verified"] = True
    target = (boxes[0].low + boxes[0].high) / 2
    target[2] = 0.075
    plan = RoutePlanner().plan(
        arm, np.array(data["start"]), target, config, boxes, "rrt", 1, 1
    )
    return data, config, arm, boxes, plan


def test_override_keeps_collision_prediction_and_can_execute_with_fake_motors():
    _, config, _, _, plan = override_plan()
    info = route_info(plan)
    assert info["allow_object_collisions"] and info["collision_predictions"]
    bus, clock = Bus(config), Clock()
    _, result = execute_target(
        bus,
        plan.target_m,
        config,
        lambda: False,
        clock=clock.now,
        sleep=clock.sleep,
        selected_plan=plan,
    )
    assert result["completed"] and bus.writes
    config["allow_object_collisions"] = False
    untouched = Bus(config)
    with pytest.raises(ValueError, match="override changed"):
        execute_target(
            untouched,
            plan.target_m,
            config,
            lambda: False,
            clock=clock.now,
            sleep=clock.sleep,
            selected_plan=plan,
        )
    assert not untouched.writes
    import time

    require_same_scene(
        info,
        dict(
            unix_time_s=time.time(),
            obstacle_detection=dict(scene_revision=999),
            obstacle_boxes=[],
        ),
    )
    stopped, stop_clock = Bus(config), Clock()
    config["allow_object_collisions"] = True
    with pytest.raises(RuntimeError, match="stop"):
        execute_target(
            stopped,
            plan.target_m,
            config,
            lambda: len(stopped.writes) > 3,
            clock=stop_clock.now,
            sleep=stop_clock.sleep,
            selected_plan=plan,
        )
    assert stopped.enabled


def test_override_does_not_remove_workspace_checks():
    data, config, arm, boxes, plan = override_plan()
    target = plan.target_m.copy()
    target[0] = config["workspace_max_m"][0] + 0.02
    with pytest.raises(ValueError):
        RoutePlanner().plan(
            arm, np.array(data["start"]), target, config, boxes, "direct", 1, 1
        )


def test_scene_renderer_and_orbit_controls_use_same_model_geometry():
    data, config, arm, boxes, plan = override_plan()
    snapshot = dict(arm=arm, q=data["start"], config=config, boxes=boxes, plan=plan)
    view = OrbitView()
    first = np.array(render_scene(snapshot, view, (800, 600)))
    view.rotate(100, 30)
    second = np.array(render_scene(snapshot, view, (800, 600)))
    assert not np.array_equal(first, second)
    view.pan += [20, -10]
    view.scroll(120)
    assert view.zoom > 1 and np.array_equal(view.pan, [20, -10])
    view.scroll(100000)
    assert view.zoom == 8
