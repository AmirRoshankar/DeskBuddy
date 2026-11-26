from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from deskbuddy.config import DEFAULT
from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.multiview import (
    calibrate_views,
    calibration_current,
    intrinsic_from_paper,
)
from deskbuddy.overlays import project
from deskbuddy.trajectory.multiview_evidence import confirm_box
from deskbuddy.trajectory.perception import DetectedBox


def synthetic_pair():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = T_BASE_CAMERA.tolist()
    base = np.array(
        [[0.15, -0.1, 0], [0.36, -0.1, 0], [0.36, 0.197, 0], [0.15, 0.197, 0]]
    )
    c["test_bed"] = dict(corners_base_m=base.tolist(), inset_m=0.01)
    primary_intr = dict(fx=600, fy=600, ppx=320, ppy=240)
    camera = np.eye(4)
    camera[:3, 3] = [0.52, -0.28, 0.35]
    z = np.array([0.255, 0.0485, 0]) - camera[:3, 3]
    z /= np.linalg.norm(z)
    right = np.cross(z, [0, 0, 1])
    right /= np.linalg.norm(right)
    camera[:3, :3] = np.column_stack((right, np.cross(z, right), z))
    second_intr = dict(fx=900, fy=900, ppx=640, ppy=360)
    pp = [project(p, c["T_base_camera"], primary_intr) for p in base]
    sp = [project(p, camera, second_intr) for p in base]
    images = []
    for size, pose, intr, corners in [
        ((640, 480), T_BASE_CAMERA, primary_intr, pp),
        ((1280, 720), camera, second_intr, sp),
    ]:
        image = Image.new("RGB", size, "#513b23")
        draw = ImageDraw.Draw(image)
        draw.polygon([tuple(p) for p in corners], fill="white")
        for a, b in [
            (base[0] * 0.5 + base[1] * 0.5, base[3] * 0.5 + base[2] * 0.5),
            (base[0] * 0.5 + base[3] * 0.5, base[1] * 0.5 + base[2] * 0.5),
        ]:
            draw.line(
                (project(a, pose, intr), project(b, pose, intr)),
                fill="#2050ff",
                width=4,
            )
        images.append(np.array(image))
    return c, primary_intr, camera, pp, sp, images


def test_a4_alignment_recovers_camera_pose_but_does_not_claim_unverified_elevated_accuracy():
    c, intr, pose, pp, sp, images = synthetic_pair()
    result = calibrate_views(*images, pp, sp, intr, c, np.full((480, 640), 0.65))
    assert result["blue_agreement"] > 0.8
    assert not result[
        "metric_3d_validated"
    ]  # no elevated RGB-D correspondences in this fixture
    assert np.allclose(result["T_base_camera"], pose, atol=0.002)
    c["test_bed"] = result["primary_bed"]
    c["secondary_camera_calibration"] = result
    assert calibration_current(c, (1280, 720))
    c["T_base_camera"][0][3] += 0.01
    assert not calibration_current(c)


def test_frontal_paper_does_not_invent_an_observable_focal_length():
    assert (
        intrinsic_from_paper(
            np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]), (1280, 720)
        )
        is None
    )


def test_planar_confirmation_uses_contact_edge_not_raised_object_center():
    c = dict(
        H_secondary_paper=np.diag([1000.0, 1000.0, 1]).tolist(),
        corners_base_m=[[0, 0, 0], [0.21, 0, 0], [0.21, 0.297, 0], [0, 0.297, 0]],
        metric_3d_validated=False,
    )
    box = DetectedBox(
        np.array([0.08, 0.10, 0]), np.array([0.12, 0.14, 0.07]), 100, 0.07, (0, 0, 1, 1)
    )
    mask = np.zeros((300, 220), bool)
    mask[30:140, 85:115] = True  # most of this raised mask is not on the table
    assert confirm_box(box, [dict(mask=mask)], c)[0]
    wrong = np.zeros_like(mask)
    wrong[10:30, 10:30] = True
    assert not confirm_box(box, [dict(mask=wrong)], c)[0]


def test_failed_blue_check_can_be_saved_without_claiming_valid_3d():
    c, intr, pose, pp, sp, images = synthetic_pair()
    result = calibrate_views(
        *images, pp, sp[1:] + sp[:1], intr, c, np.full((480, 640), 0.85), strict=False
    )
    assert result["validation_warnings"]
    assert not result["validation_passed"]
    assert not result["metric_3d_validated"]
    assert np.isfinite(result["H_secondary_paper"]).all()
