from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deskbuddy.trajectory.rgb_can_perception import cylinder, projector
from deskbuddy.trajectory.yolo_perception import YoloDetector


def rgb_fixture():
    world = np.array([[0, 0, 0], [0.21, 0, 0], [0.21, 0.297, 0], [0, 0.297, 0]])
    k = dict(fx=420.0, fy=420.0, ppx=320.0, ppy=240.0)

    def camera(position):
        forward = np.array([0.105, 0.1485, 0]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        t = np.eye(4)
        t[:3, :3] = np.c_[right, np.cross(forward, right), forward]
        t[:3, 3] = position
        c = (world - position) @ t[:3, :3]
        pixels = np.c_[
            c[:, 0] / c[:, 2] * k["fx"] + k["ppx"],
            c[:, 1] / c[:, 2] * k["fy"] + k["ppy"],
        ]
        return t, pixels

    a, ap = camera(np.array([0.105, -0.3, 0.5]))
    b, bp = camera(np.array([0.105, 0.6, 0.5]))
    config = dict(
        task_mode="can_discard",
        secondary_camera=True,
        calibration_verified=True,
        workspace_min_m=[-0.1, -0.1, 0],
        workspace_max_m=[0.4, 0.4, 0.4],
        table_z_m=0,
        assume_level_table=True,
        T_base_camera=a.tolist(),
        test_bed=dict(
            corners_base_m=world.tolist(), corner_pixels=ap.tolist(), inset_m=0.01
        ),
        secondary_camera_calibration=dict(
            T_base_camera=b.tolist(),
            intrinsics=k,
            secondary_pixels=bp.tolist(),
            primary_pixels=ap.tolist(),
            image_size=[640, 480],
        ),
    )
    center = np.array([0.105, 0.14])
    torch = pytest.importorskip("torch")
    results = []
    for t, pixels in ((a, ap), (b, bp)):
        points = projector(t, k, world, pixels)(cylinder(center, 0))
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.int32(points)), 1)
        v, u = np.nonzero(mask)
        result = SimpleNamespace(
            boxes=[
                SimpleNamespace(
                    xyxy=torch.tensor([[u.min(), v.min(), u.max(), v.max()]]),
                    conf=torch.tensor(0.9),
                    cls=torch.tensor(0),
                )
            ],
            masks=SimpleNamespace(data=torch.tensor(mask[None])),
            names={0: "can"},
        )
        results.append(result)
    return config, k, results, center


def test_can_detection_batched_rgb_and_independent_of_depth(monkeypatch):
    import time

    import deskbuddy.multiview as multiview
    import deskbuddy.trajectory.yolo_perception as yolo

    config, intrinsics, results, center = rgb_fixture()
    monkeypatch.setattr(multiview, "calibration_current", lambda *a: True)
    monkeypatch.setattr(multiview, "blue_agreement", lambda *a: None)

    def forbidden(*a, **kw):
        raise AssertionError("Depth geometry must not run for RGB can detection")

    monkeypatch.setattr(yolo, "paper_surface", forbidden)
    calls = []

    def predict(inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        assert all(image.ndim == 3 and image.shape[-1] == 3 for image in inputs)
        calls.append(inputs)
        return results

    detector = YoloDetector(SimpleNamespace(predict=predict))
    rgb = np.zeros((480, 640, 3), np.uint8)
    geometries = []
    for depth in (
        np.zeros((480, 640)),
        np.full((480, 640), np.nan),
        np.ones((480, 640)) * 100,
    ):
        boxes, report = detector.detect(
            rgb, depth, intrinsics, config, secondary=(rgb, time.time())
        )
        assert len(boxes) == 1
        np.testing.assert_allclose(
            (boxes[0].low[:2] + boxes[0].high[:2]) / 2, center, atol=0.002
        )
        assert "unused" in report["depth_role"]
        assert report["model"] == "YOLOE-26s-seg-pf"
        geometries.append(np.r_[boxes[0].low, boxes[0].high])
    np.testing.assert_array_equal(geometries[0], geometries[1])
    np.testing.assert_array_equal(geometries[1], geometries[2])
    assert len(calls) == 3


def test_can_requires_fresh_calibrated_second_view(monkeypatch):
    config, intrinsics, results, _ = rgb_fixture()
    detector = YoloDetector(SimpleNamespace(predict=lambda *a, **kw: results[:1]))
    boxes, report = detector.detect(
        np.zeros((480, 640, 3), np.uint8), None, intrinsics, config
    )
    assert boxes == []
    assert "Awaiting" in report["geometry_status"]


def test_projected_arm_mask_cannot_erase_a_strong_two_view_can():
    from deskbuddy.trajectory.multiview_evidence import instances_from_result
    from deskbuddy.trajectory.rgb_can_perception import detect_rgb_cans

    config, k, results, center = rgb_fixture()
    boxes, report = detect_rgb_cans(
        instances_from_result(results[0]),
        instances_from_result(results[1]),
        config,
        config["secondary_camera_calibration"],
        np.ones((480, 640), bool),
        k,
    )
    assert len(boxes) == 1
    np.testing.assert_allclose(
        (boxes[0].low[:2] + boxes[0].high[:2]) / 2, center, atol=0.002
    )
