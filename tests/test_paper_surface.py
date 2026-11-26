from types import SimpleNamespace

import numpy as np
from test_obstacle_preview import depth_fixture
from test_yolo_perception import fake_model

from deskbuddy.trajectory.paper_surface import paper_surface
from deskbuddy.trajectory.yolo_perception import YoloDetector


def calibrated_paper():
    config, intrinsics, _ = depth_fixture()
    config["test_bed"] = dict(
        corners_base_m=[
            [0, 0, -0.015],
            [0.21, 0, 0.015],
            [0.21, 0.297, 0.015],
            [0, 0.297, -0.015],
        ],
        inset_m=0.02,
        clearance_margin_m=0.003,
    )
    corners = np.array(config["test_bed"]["corners_base_m"])
    normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    transform = np.array(config["T_base_camera"])
    v, u = np.indices((200, 200))
    rays = np.stack(
        (
            (u - intrinsics["ppx"]) / intrinsics["fx"],
            (v - intrinsics["ppy"]) / intrinsics["fy"],
            np.ones_like(u),
        ),
        axis=-1,
    )
    depth = ((corners[0] - transform[:3, 3]) @ normal) / (
        rays @ transform[:3, :3].T @ normal
    )
    rgb = np.full((200, 200, 3), 220, dtype=np.uint8)
    return config, intrinsics, rgb, depth


def test_known_tilted_paper_with_noise_and_bias_is_excluded():
    config, intrinsics, rgb, depth = calibrated_paper()
    rng = np.random.default_rng(10)
    depth += rng.normal(0, 0.0015, depth.shape) - 0.01
    depth -= (np.indices(depth.shape)[1] - 100) * 0.00015  # residual depth-plane tilt
    flat, footprint, bias, report = paper_surface(rgb, depth, intrinsics, config)
    assert report["reference_spread"]
    assert 0.004 < np.median(bias[footprint]) < 0.010
    assert flat[footprint].mean() > 0.98
    # Both SDK-style intrinsics and plain dictionaries are supported.
    other = paper_surface(rgb, depth, SimpleNamespace(**intrinsics), config)
    assert np.array_equal(flat, other[0])
    detector = YoloDetector(fake_model(footprint.astype(float), (60, 45, 140, 155)))
    boxes, report = detector.detect(rgb, depth, intrinsics, config)
    assert not boxes
    assert report["paper_surface"]["rejected_instances"] == 1


def test_raised_object_on_paper_survives_and_merged_box_is_localized():
    config, intrinsics, rgb, depth = calibrated_paper()
    _, footprint, _, _ = paper_surface(rgb, depth, intrinsics, config)
    depth[88:112, 88:112] -= 0.10
    flat, _, _, _ = paper_surface(rgb, depth, intrinsics, config)
    assert not flat[88:112, 88:112].any()
    detector = YoloDetector(fake_model(footprint.astype(float), (60, 45, 140, 155)))
    boxes, _ = detector.detect(rgb, depth, intrinsics, config)
    assert len(boxes) == 1 and boxes[0].height_m > 0.08
    x1, y1, x2, y2 = boxes[0].pixel_roi
    assert x2 - x1 < 30 and y2 - y1 < 30


def test_paper_depth_holes_are_not_fabricated_and_tall_white_object_cannot_refit_plane():
    config, intrinsics, rgb, depth = calibrated_paper()
    _, footprint, _, _ = paper_surface(rgb, depth, intrinsics, config)
    depth[footprint] -= 0.1
    flat, _, bias, report = paper_surface(rgb, depth, intrinsics, config)
    assert not flat[footprint].any()
    assert bias == 0 and not report["reference_spread"]
    depth[:] = 0
    assert not paper_surface(rgb, depth, intrinsics, config)[0].any()
