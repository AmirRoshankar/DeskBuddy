from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deskbuddy.config import DEFAULT
from deskbuddy.grasp_geometry import (
    attach_grasp_preview,
    grasp_settings,
    oriented_ik,
    table_normal,
    tilt_degrees,
    tool_geometry,
)
from deskbuddy.kinematics import Arm
from deskbuddy.planning import Plan
from deskbuddy.test_bed import table_clearance
from deskbuddy.trajectory.execution import lift_route, route_info


def fixture(tilted=False):
    config = deepcopy(DEFAULT)
    rotation = Rotation.from_euler("x", 4 if tilted else 0, degrees=True).as_matrix()
    bed = np.array([[0, 0, 0], [0.21, 0, 0], [0.21, 0.297, 0], [0, 0.297, 0]])
    bed = bed @ rotation.T + [0.13, -0.11, -0.005]
    config["test_bed"] = dict(corners_base_m=bed.tolist(), inset_m=0.005)
    config["workspace_min_m"] = [0.1, -0.2, -0.03]
    config["workspace_max_m"] = [0.45, 0.25, 0.30]
    config["tool_offset_m"] = [0.0127, 0.0031, 0.0005]
    arm = Arm(tool_offset_m=config["tool_offset_m"])
    seed = np.deg2rad(config["preview_start_deg"])
    source = bed[0] + 0.5 * (bed[1] - bed[0]) + 0.5 * (bed[3] - bed[0])
    return config, arm, seed, source


@pytest.mark.parametrize("tilted", [False, True])
def test_grasp_keeps_calibrated_tip_at_key_and_physical_tool_normal_to_table(tilted):
    config, arm, seed, source = fixture(tilted)
    normal, contact, above, aligned, touching = grasp_settings(
        arm, config, source, source + [0, 0, 0.10]
    )
    q = oriented_ik(arm, contact, seed, -normal, thorough=True)
    tip, wrist, _ = tool_geometry(arm, q)
    assert np.linalg.norm(tip - contact) < 0.0005
    assert tilt_degrees(arm, q, normal) < 0.2
    assert 0.095 < (wrist - tip) @ normal < 0.105
    assert abs((tip - source) @ normal - 0.005) < 0.0005
    assert np.min(table_clearance(arm.frames(q)[0][2:], touching)) > 0
    assert "wrist_roll" not in aligned["locked_joints"]
    assert config["locked_joints"] == DEFAULT["locked_joints"]
    approach = oriented_ik(arm, above, q, -normal, thorough=True)
    # TCP accuracy alone does not prove the physical fingers clear the plane.
    # This calibrated-offset fixture exposes a sub-mm body overlap at contact.
    with pytest.raises(ValueError, match="table clearance"):
        lift_route(
            arm,
            approach,
            contact,
            np.array([arm.fk(approach)[:2], contact[:2]]),
            touching,
            method="grasp descent",
            candidates=1,
            boxes=[],
            scene_revision=0,
        )
    touching["allow_object_collisions"] = True
    descent = lift_route(
        arm,
        approach,
        contact,
        np.array([arm.fk(approach)[:2], contact[:2]]),
        touching,
        method="grasp descent",
        candidates=1,
        boxes=[],
        scene_revision=0,
    )
    assert (
        route_info(descent)["max_grasp_tilt_deg"]
        <= tilt_degrees(arm, approach, normal) + 0.3
    )
    assert route_info(descent)["grasp_tilt_deg"] < 0.2
    assert any(i.get("table_collision_prediction") for i in descent.planner_info)


def test_far_target_uses_best_feasible_tilt_without_sacrificing_tip_position():
    arm = Arm()
    seed = np.deg2rad(DEFAULT["preview_start_deg"])
    target = np.array([0.32, 0.08, 0.115])
    position_only = arm.ik(target, seed, 0.002, [])
    q = oriented_ik(arm, target, seed, [0, 0, -1], thorough=True)
    assert np.linalg.norm(arm.fk(q) - target) < 0.0005
    assert 1 < tilt_degrees(arm, q, [0, 0, 1]) < 25
    assert tilt_degrees(arm, q, [0, 0, 1]) < tilt_degrees(arm, position_only, [0, 0, 1])
    assert (q >= arm.lower - 1e-8).all() and (q <= arm.upper + 1e-8).all()


def test_preview_carries_actual_wrist_contact_pose_for_3d_inspection():
    config, arm, seed, source = fixture(True)
    target = source + [0, 0, 0.10]
    q = arm.ik(target, seed, 0.002, [])
    plan = Plan(np.array([0.0, 1.0]), np.array([seed, q]), target, 0.0)
    attach_grasp_preview(plan, arm, config, source)
    info = plan.planner_info[-1]
    assert info["tilt_deg"] < 0.2
    assert np.allclose(info["table_normal"], table_normal(config))
    assert np.allclose(arm.fk(info["grasp_pose_rad"]), info["contact_m"], atol=0.0005)


def test_unreachable_grasp_rejects_instead_of_moving_tip_to_improve_orientation():
    with pytest.raises(ValueError, match="No downward grasp pose"):
        oriented_ik(Arm(), [2, 0, 0], np.zeros(5), [0, 0, -1])


def test_grasp_cannot_lower_operator_selected_z_floor():
    config, arm, seed, source = fixture()
    config["workspace_min_m"][2] = 0.08
    _, contact, above, aligned, touching = grasp_settings(
        arm, config, source, source + [0, 0, 0.15]
    )
    assert contact[2] >= 0.08 - 1e-10
    assert touching["workspace_min_m"][2] == 0.08
    assert above[2] >= contact[2]
