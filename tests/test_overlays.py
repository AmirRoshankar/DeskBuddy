from copy import deepcopy

import numpy as np
from PIL import Image

from deskbuddy.config import DEFAULT
from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.kinematics import Arm
from deskbuddy.overlays import (
    draw_overlays,
    estimated_goal_check,
    grid_segments,
    project,
)

INTR = {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240}


def config():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = T_BASE_CAMERA.tolist()
    return c


def test_projection_and_grid_have_lines_without_numbered_targets():
    c = config()
    assert np.allclose(project([0.25, 0, 0], c["T_base_camera"], INTR), [320, 240])
    lines = grid_segments(c, INTR, divisions=4)
    assert len(lines) == 10
    assert sum(border for _, _, border in lines) == 4


def test_pixel_overlap_alone_does_not_establish_3d_goal_reached():
    c = config()
    result = estimated_goal_check([0.25, 0, 0.1], [0.25, 0, 0.2], c, INTR)
    assert result["projected_overlap"]
    assert result["pixel_gap"] == 0
    assert result["model_gap_mm"] == 100
    assert not result["estimated_goal_reached"]
    assert not result["independent_visual_measurement"]


def test_tip_marker_uses_measured_pose_and_distinguishes_surface_from_hover():
    c = config()
    arm = Arm()
    q = np.deg2rad(c["preview_start_deg"])
    target = arm.fk(q)
    surface = target - [0, 0, 0.06]
    image = Image.new("RGB", (640, 480))
    result = draw_overlays(image, arm, q, target, surface, c, INTR)
    assert np.allclose(result["tip_base_m"], arm.fk(q))
    assert result["estimated_goal_reached"]
    assert result["pixel_gap"] == 0
    assert project(surface, c["T_base_camera"], INTR) != project(
        target, c["T_base_camera"], INTR
    )


def test_plan_overlay_uses_full_fk_trajectory_and_keeps_measured_arm_independent():
    from deskbuddy.motion_overlay import motion_geometry
    from deskbuddy.planning import plan_reach

    c = config()
    arm = Arm()
    start = np.deg2rad(c["preview_start_deg"])
    target = arm.fk(start + np.deg2rad([5, 3, -2, 1, 0]))
    plan = plan_reach(arm, start, target, c)
    current, path, goal = motion_geometry(arm, start, plan)
    assert np.allclose(current, arm.frames(start)[0])
    assert np.allclose(path[0], arm.fk(start))
    assert np.allclose(path[-1], arm.fk(plan.joints_rad[-1]))
    assert np.allclose(goal, arm.frames(plan.joints_rad[-1])[0])
    changed = start + np.deg2rad([1, 0, 0, 0, 0])
    measured, same_path, _ = motion_geometry(arm, changed, plan)
    assert same_path is path  # Immutable preview geometry is cached for both views.
    assert not np.allclose(measured, current)
    image = Image.new("RGB", (640, 480))
    result = draw_overlays(image, arm, changed, target, None, c, INTR, plan=plan)
    assert np.allclose(result["tip_base_m"], arm.fk(changed))
