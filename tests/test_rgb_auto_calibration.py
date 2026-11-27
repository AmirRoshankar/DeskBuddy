import numpy as np
from test_rgb_hand_eye import look_at, project

from deskbuddy.kinematics import Arm
from deskbuddy.rgb_auto_calibration import fit_rgb_marker


def test_rgb_fit_without_verified_stereo_or_depth():
    arm = Arm()
    joints = np.random.default_rng(21).uniform(
        np.deg2rad([-30, -35, 10, -20, -20]), np.deg2rad([30, 30, 65, 40, 20]), (15, 5)
    )
    cameras = []
    for position in ([0.1, 0.6, 0.6], [0.4, -0.6, 0.6]):
        pose = look_at(np.array(position), [0.2, 0, 0.15])
        cameras.append(
            dict(
                T_base_camera=pose.tolist(),
                intrinsics=dict(fx=600, fy=600, ppx=320, ppy=240),
            )
        )
    world = []
    for q in joints:
        frame = arm.frames(q)[1]
        world.append(frame[:3, 3] + frame[:3, :3] @ np.array([0.01, 0.005, 0]))
    k = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    pixels = np.stack(
        [project(np.asarray(c["T_base_camera"]), k, world) for c in cameras], axis=1
    )
    cameras[1]["T_base_camera"][0][3] += 0.01
    result = fit_rgb_marker(arm, joints, pixels, cameras)
    assert result["passed"]
    assert result["depth_independent"]
    assert not result["metric_accuracy_verified"]
    assert np.max(result["heldout_error_px"]) < 1


def test_webcam_only_fit_accepts_missing_auxiliary_observations():
    arm = Arm()
    joints = np.random.default_rng(22).uniform(
        np.deg2rad([-25, -30, 10, -20, -20]), np.deg2rad([25, 30, 65, 40, 20]), (15, 5)
    )
    cameras = []
    for position in ([0.1, 0.6, 0.6], [0.4, -0.6, 0.6]):
        pose = look_at(np.array(position), [0.2, 0, 0.15])
        cameras.append(
            dict(
                T_base_camera=pose.tolist(),
                intrinsics=dict(fx=600, fy=600, ppx=320, ppy=240),
            )
        )
    world = [arm.fk(q) for q in joints]
    k = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    pixels = np.stack(
        [project(np.asarray(c["T_base_camera"]), k, world) for c in cameras], axis=1
    )
    pixels[:, 0] = np.nan
    result = fit_rgb_marker(arm, joints, pixels, cameras)
    assert result["passed"]
    assert result["auxiliary_samples"] == 0
    assert max(result["webcam_heldout_error_px"]) < 1
    assert all(row[0] is None for row in result["heldout_error_px"])
    import json

    json.dumps(result, allow_nan=False)
