import numpy as np
import pytest

from deskbuddy.orange_calibration import (
    detect_orange_marker,
    fit_arm_marker,
    triangulate_marker,
)


def test_marker_detection_rejects_ambiguous_background_and_tracks_prior():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[80:100, 60:80] = [255, 100, 0]
    marker = detect_orange_marker(image)
    np.testing.assert_allclose(marker["pixel"], [69.5, 89.5])
    image[120:140, 220:240] = [255, 100, 0]
    assert detect_orange_marker(image) is None
    assert detect_orange_marker(image, previous=[70, 90])["pixel"] == marker["pixel"]


def test_two_cameras_recover_marker_and_reject_parallel_rays():
    intr = dict(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    first = dict(intrinsics=intr, T_base_camera=np.eye(4).tolist())
    transform = np.eye(4)
    transform[0, 3] = 0.2
    second = dict(intrinsics=intr, T_base_camera=transform.tolist())
    result = triangulate_marker([320, 240], [200, 240], first, second)
    np.testing.assert_allclose(result["point_m"], [0, 0, 1], atol=1e-7)
    with pytest.raises(ValueError, match="parallel"):
        triangulate_marker([320, 240], [320, 240], first, second)


def test_marker_fit_uses_independent_holdout_and_rejects_degenerate_data():
    from deskbuddy.kinematics import Arm

    arm = Arm()
    rng = np.random.default_rng(33)
    joints = rng.uniform(
        np.deg2rad([-35, -50, -50, -40, -50]),
        np.deg2rad([35, 30, 70, 60, 50]),
        size=(18, 5),
    )
    marker_offset = np.array([0.006, -0.004, 0.012])
    points = []
    for q in joints:
        _, transform = arm.frames(q)
        points.append(transform[:3, 3] + transform[:3, :3] @ marker_offset)
    result = fit_arm_marker(arm, joints, points)
    assert result["passed"]
    assert max(result["heldout_error_mm"]) < 2
    corrupted = np.array(points)
    corrupted[-1] += [0.04, 0, 0]
    assert not fit_arm_marker(arm, joints, corrupted)["passed"]
    with pytest.raises(ValueError, match="variation"):
        fit_arm_marker(arm, joints, np.zeros((18, 3)))


def test_sheet_center_prior_selects_tip_among_other_orange_regions():
    image = np.zeros((300, 400, 3), np.uint8)
    image[100:115, 195:210] = [255, 100, 0]
    image[200:215, 40:55] = [255, 100, 0]
    image[110:125, 230:245] = [255, 100, 0]
    tip = detect_orange_marker(image, initial_hint=[[200, 90], [200, 110], [200, 130]])
    assert np.linalg.norm(np.asarray(tip["pixel"]) - [202, 107]) < 1
    tracked = detect_orange_marker(image, previous=tip["pixel"])
    assert tracked["pixel"] == tip["pixel"]


def test_optical_flow_preserves_marker_identity_with_competing_orange_patch():
    import cv2

    from deskbuddy.orange_calibration import track_orange_marker

    image = np.zeros((160, 240, 3), np.uint8)
    cv2.circle(image, (80, 80), 12, (255, 100, 0), -1)
    cv2.circle(image, (150, 100), 12, (255, 100, 0), -1)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 4]]), (240, 160))
    marker = track_orange_marker(image, shifted, [80, 80])
    assert marker is not None
    assert np.linalg.norm(np.asarray(marker["pixel"]) - [88, 84]) < 1
