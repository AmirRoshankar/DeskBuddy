import numpy as np
import pytest

from deskbuddy import rgb_hand_eye as he
from deskbuddy.kinematics import Arm


def look_at(position, target):
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    t = np.eye(4)
    t[:3, :3] = np.c_[right, np.cross(forward, right), forward]
    t[:3, 3] = position
    return t


def project(t, k, points):
    camera = (np.asarray(points) - t[:3, 3]) @ t[:3, :3]
    uv = camera @ k.T
    return uv[:, :2] / uv[:, 2:]


def test_webcam_pose_and_focal_from_arm_clicks_then_realsense_from_paper():
    pytest.importorskip("cv2")
    arm = Arm()
    rng = np.random.default_rng(0)
    size = (1280, 720)
    web = look_at(np.array([0.25, 0.55, 0.45]), [0.22, 0.0, 0.05])
    k_web = he.camera_matrix(900.0, size)
    records = []
    while len(records) < 10:
        q = rng.uniform(
            np.deg2rad([-40, -40, -10, -30, 0]), np.deg2rad([40, 30, 80, 60, 0])
        )
        tip = arm.fk(q)
        if not 0.03 < tip[2] < 0.3:
            continue
        pixel = project(web, k_web, tip[None])[0] + rng.normal(0, 0.5, 2)
        records.append(dict(pixel=pixel.tolist(), base_point_model_m=tip.tolist()))
    fit = he.fit_camera(records, size)
    assert fit["passed"]
    assert fit["intrinsics"]["fx"] == pytest.approx(900, rel=0.05)
    assert np.allclose(np.asarray(fit["T_base_camera"])[:3, 3], web[:3, 3], atol=0.02)

    paper = np.array(
        [[0.14, -0.15, 0], [0.35, -0.15, 0], [0.35, 0.147, 0], [0.14, 0.147, 0]]
    )
    realsense = look_at(np.array([0.19, -0.39, 0.43]), [0.25, 0.0, 0.0])
    rs_k = dict(fx=615.0, fy=614.0, ppx=308.0, ppy=244.0)
    k_rs = np.array([[615.0, 0, 308.0], [0, 614.0, 244.0], [0, 0, 1]])
    t_rs, corners, secondary = he.register_from_paper(
        fit,
        project(web, k_web, paper),
        project(realsense, k_rs, paper),
        rs_k,
        size,
        0.0,
    )
    assert np.allclose(corners, paper, atol=0.01)
    assert np.allclose(t_rs[:3, 3], realsense[:3, 3], atol=0.03)
    assert secondary["metric_3d_validated"] and secondary["validation_passed"]


def test_clicks_need_spread_and_eight_points():
    with pytest.raises(ValueError, match="eight"):
        he.fit_camera(
            [dict(pixel=[0, 0], base_point_model_m=[0, 0, 0])] * 3, (640, 480)
        )


def test_rejects_duplicate_and_outside_image_points():
    rows = [dict(pixel=[200, 200], base_point_model_m=[0.1, 0.1, 0.1])] * 8
    with pytest.raises(ValueError, match="too similar"):
        he.fit_camera(rows, (640, 480))
    rows = [
        dict(pixel=[-1, 200], base_point_model_m=[i * 0.01, 0, 0]) for i in range(8)
    ]
    with pytest.raises(ValueError, match="inside the camera"):
        he.fit_camera(rows, (640, 480))
