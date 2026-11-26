import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deskbuddy.kinematics import Arm
from deskbuddy.tool_point_fit import check_tool_point, fit_tool_point


def test_configured_tip_is_used_by_fk_and_ik_without_changing_link_origins():
    original = Arm()
    offset = np.array([0.013, 0.003, 0])
    corrected = Arm(tool_offset_m=offset)
    q = np.deg2rad([5, -15, 30, 15, 0])
    points, frame = original.frames(q)
    changed, tip_frame = corrected.frames(q)
    assert np.allclose(changed[:-1], points)
    expected = frame[:3, 3] + frame[:3, :3] @ offset
    assert np.allclose(corrected.fk(q), expected)
    assert np.allclose(tip_frame[:3, 3], expected)
    goal = corrected.fk(q + np.deg2rad([3, 2, -1, 2, 0]))
    result = corrected.ik(goal, q, 0.002)
    assert np.linalg.norm(corrected.fk(result) - goal) < 0.002


def test_tool_point_and_camera_recovered_with_frozen_joints_and_fresh_validation():
    arm = Arm()
    offset = np.array([0.013, 0.003, 0])
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    translation = np.array([0.1, 0.2, -0.3])
    rows = []
    for pose in np.random.default_rng(22).uniform(-0.8, 0.8, (10, 5)):
        frame = arm.frames(pose)[1]
        point = frame[:3, 3] + frame[:3, :3] @ offset
        rows.append(
            {
                "session_id": "test",
                "camera_serial": "cam",
                "unix_time_s": 1,
                "q_model_rad": pose.tolist(),
                "camera_point_m": ((point - translation) @ rotation).tolist(),
            }
        )
    candidate = fit_tool_point(rows[:8])
    assert candidate["passed_initial_checks"]
    assert np.allclose(candidate["tool_offset_m"], offset, atol=1e-6)
    assert not candidate["joint_offsets_changed"]
    for row in rows[8:]:
        row["unix_time_s"] = candidate["created_unix_s"] + 1
    result = check_tool_point(candidate, rows)
    assert result["passed"] and not result["refitted_on_checks"]
    rows[-1]["camera_point_m"][0] += 0.04
    assert not check_tool_point(candidate, rows)["passed"]
    rows[0]["camera_point_m"][0] += 0.01
    with pytest.raises(ValueError, match="changed"):
        check_tool_point(candidate, rows)
