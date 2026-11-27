import pytest
from test_multiview import synthetic_pair

from deskbuddy.multiview import calibrate_views
from deskbuddy.overlays import project
from deskbuddy.rgb_stereo_alignment import refine_rgb_pairs


def fixture():
    c, intr, pose, pp, sp, images = synthetic_pair()
    result = calibrate_views(*images, pp, sp, intr, c, None)
    points = [
        [0.19, -0.03, 0.07],
        [0.28, 0.10, 0.12],
        [0.32, -0.03, 0.08],
        [0.23, 0.06, 0.16],
        [0.26, 0.03, 0.10],
        [0.29, 0.08, 0.14],
    ]
    observations = [
        dict(
            primary_pixel=project(p, c["T_base_camera"], intr),
            secondary_pixel=project(p, pose, result["intrinsics"]),
        )
        for p in points
    ]
    return c, intr, result, observations


def test_rgb_refines_pose_with_no_depth_and_heldout_checks():
    c, intr, result, observations = fixture()
    result["T_base_camera"][0][3] += 0.015
    check = refine_rgb_pairs(result, observations, intr, c)
    assert check["passed"]
    assert check["depth_independent"]
    assert max(check["error_px"]) < 0.5


def test_wrong_heldout_correspondence_fails():
    c, intr, result, observations = fixture()
    observations[-1]["secondary_pixel"] = [10, 10]
    with pytest.raises(ValueError, match="Held-out RGB P6"):
        refine_rgb_pairs(result, observations, intr, c)
