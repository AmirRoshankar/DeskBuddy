from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.control_plan import preview_from_feedback
from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.kinematics import JOINTS, Arm
from deskbuddy.overlays import project
from deskbuddy.workspace_geometry import box_handles, drag_box, target_guides

INTRINSICS = dict(fx=600, fy=600, ppx=320, ppy=240)
CAMERA = np.array(T_BASE_CAMERA, float)
CAMERA[0, 3] += 0.2


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [0, 1])
def test_perspective_face_drag_changes_only_the_selected_bound(axis, side):
    config = deepcopy(DEFAULT)
    config["T_base_camera"] = CAMERA.tolist()
    handle = next(h for h in box_handles(config) if h[2:] == (axis, side))
    moved = handle[1].copy()
    moved[axis] += 0.015
    begin = project(handle[1], CAMERA, INTRINSICS)
    end = project(moved, CAMERA, INTRINSICS)
    result = drag_box(config, handle, begin, end, INTRINSICS)
    key = "workspace_min_m" if side == 0 else "workspace_max_m"
    expected = np.array(config[key])
    expected[axis] += 0.015
    assert np.allclose(result[key], expected)
    other = "workspace_max_m" if side == 0 else "workspace_min_m"
    assert result[other] == config[other]


def test_center_drag_translates_box_in_xy_without_changing_size_or_height():
    config = deepcopy(DEFAULT)
    config["T_base_camera"] = CAMERA.tolist()
    handle = box_handles(config)[0]
    delta = np.array([0.02, -0.01, 0])
    result = drag_box(
        config,
        handle,
        project(handle[1], CAMERA, INTRINSICS),
        project(handle[1] + delta, CAMERA, INTRINSICS),
        INTRINSICS,
    )
    for key in ("workspace_min_m", "workspace_max_m"):
        assert np.allclose(result[key], np.array(config[key]) + delta)


def test_target_guides_are_three_perpendicular_face_projections():
    config = deepcopy(DEFAULT)
    target = np.array([0.2, -0.08, 0.17])
    guides = target_guides(target, config)
    assert len(guides) == 3
    for axis, (start, end) in enumerate(guides):
        assert np.array_equal(start, target)
        assert np.array_equal(np.delete(end, axis), np.delete(target, axis))
        assert end[axis] in (
            config["workspace_min_m"][axis],
            config["workspace_max_m"][axis],
        )


def test_power_reset_goal_registers_do_not_block_torque_off_preview():
    config = deepcopy(DEFAULT)
    arm = Arm()
    measured = np.deg2rad(config["preview_start_deg"])
    target = arm.fk(measured + np.deg2rad([2, 3, -2, 1, 0]))
    state = dict(
        q_goal_model_rad=np.deg2rad([-173, -184, -175, -177, -180]),
        torque_enabled={j: 0 for j in JOINTS},
    )
    plan = preview_from_feedback(arm, measured, state, target, config)
    assert np.allclose(plan.joints_rad[0], measured)
    assert plan.endpoint_error_m < config["ik_tolerance_m"]
    # The same invalid goal must still be rejected if motors really hold it.
    state["torque_enabled"] = {j: 1 for j in JOINTS}
    with pytest.raises(ValueError, match="Existing hold target outside model limits"):
        preview_from_feedback(arm, measured, state, target, config)
