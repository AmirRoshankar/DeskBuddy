"""Frozen desk failure: a tip-only route passed XY but hit gripper clearance."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.execution import (
    ArmCollision,
    RoutePlanner,
    check_pose_obstacles,
    execution_scene,
    lift_route,
    route_info,
    segment_box_distance,
)
from deskbuddy.trajectory.perception import DetectedBox
from deskbuddy.trajectory.route_refinement import refine_route


def replay():
    data = json.loads(
        (Path(__file__).parent / "fixtures/clearance_mismatch.json").read_text()
    )
    config = deepcopy(DEFAULT)
    config.update(data["config"])
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    boxes = [
        DetectedBox(
            np.array(b["low_m"]),
            np.array(b["high_m"]),
            b["pixels"],
            b["height_above_table_m"],
            tuple(b["pixel_roi"]),
        )
        for b in data["boxes"]
    ]
    return data, config, arm, boxes


def test_capsule_box_corner_uses_euclidean_not_cubic_inflation():
    class ArmAtCorner:
        def frames(self, pose):
            return np.array([[0.05, 0.05, -0.02], [0.05, 0.05, 0.02]]), None

    box = dict(low_m=[0, 0, 0], high_m=[0.01, 0.01, 0.01])
    check_pose_obstacles(ArmAtCorner(), [0], [box])
    assert segment_box_distance(
        np.array([0.05, 0.05, -0.02]),
        np.array([0.05, 0.05, 0.02]),
        np.zeros(3),
        np.full(3, 0.01),
    ) == pytest.approx(np.sqrt(2) * 0.04)


def test_rrt_replay_now_plans_for_the_gripper_before_lifting():
    data, config, arm, boxes = replay()
    plan = RoutePlanner().plan(
        arm,
        np.array(data["start"]),
        np.array(data["target"]),
        config,
        boxes,
        "rrt",
        4,
        1,
    )
    info = route_info(plan)
    assert info["planning_space"] == "joint"
    assert info["collision_state_checks"] > 0
    assert info["trajectory_method"] == "rrt" and not info["fallback"]


def test_frozen_imf_route_is_refined_without_rrt_or_weight_loading(monkeypatch):
    data, config, arm, boxes = replay()
    raw = np.array(
        data["imf_raw_paths"][2]
    )  # Recorded third proposal, not a new model draw.
    kwargs = dict(method="imf", candidates=4, boxes=boxes, scene_revision=1)
    # Reproduce the original failure under the physical geometry checker.
    with pytest.raises(ArmCollision) as error:
        lift_route(arm, data["start"], data["target"], raw, config, **kwargs)
    assert error.value.link >= 4 and error.value.path_fraction is not None

    def forbidden(*args, **kwargs):
        raise AssertionError("Local refinement must not substitute RRT")

    monkeypatch.setattr("deskbuddy.trajectory.rrt.rrt_connect", forbidden)
    scene = execution_scene(config, boxes, arm)
    refined, report = refine_route(raw, scene)
    assert report["applied"] and report["max_shift_m"] <= 0.06501
    assert np.array_equal(refined[[0, -1]], raw[[0, -1]])
    plan = lift_route(arm, data["start"], data["target"], refined, config, **kwargs)
    assert route_info(plan)["trajectory_method"] == "imf"


def test_local_repair_respects_budget_and_rejects_blocked_endpoints():
    from deskbuddy.trajectory.collision import Scene

    scene = Scene(boxes=((0.09, 0.12, 0.12, 0.17),))
    raw = np.linspace([0.02, 0.145], [0.19, 0.145], 32)
    with pytest.raises(ValueError, match="refinement"):
        refine_route(raw, scene, max_shift=0.001)
    with pytest.raises(ValueError, match="Start or target"):
        refine_route(np.linspace([0.10, 0.14], [0.19, 0.145], 32), scene)


def test_empty_scene_can_refine_a_small_workspace_excursion():
    from deskbuddy.trajectory.collision import Scene

    scene = Scene()
    raw = np.array([[0.02, 0.1], [-0.01, 0.15], [0.19, 0.2]])
    refined, report = refine_route(raw, scene)
    assert scene.path_free(refined) and report["method"] == "workspace projection"
    assert np.array_equal(refined[[0, -1]], raw[[0, -1]])
