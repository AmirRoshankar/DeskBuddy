from types import SimpleNamespace

import numpy as np
import pytest
from test_obstacle_preview import depth_fixture

from deskbuddy.trajectory.yolo_perception import YoloDetector


def fake_model(mask, roi=(80, 75, 120, 125)):
    torch = pytest.importorskip("torch")
    detection = SimpleNamespace(
        xyxy=torch.tensor([roi]), conf=torch.tensor(0.8), cls=torch.tensor(0)
    )
    result = SimpleNamespace(
        boxes=[detection],
        masks=SimpleNamespace(data=torch.tensor(mask[None])),
        names={0: "object"},
    )
    return SimpleNamespace(predict=lambda *a, **kw: [result])


def test_rgb_paper_mask_is_ignored_but_same_shape_raised_object_is_detected():
    config, intrinsics, depth = depth_fixture()
    mask = np.zeros_like(depth)
    mask[80:120, 85:115] = 1
    detector = YoloDetector(fake_model(mask))
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    assert detector.detect(rgb, depth, intrinsics, config)[0] == []
    depth[mask.astype(bool)] = 0.59  # 1 cm paper-like surface: ignored.
    assert detector.detect(rgb, depth, intrinsics, config)[0] == []
    depth[mask.astype(bool)] = 0.5
    boxes, report = detector.detect(rgb, depth, intrinsics, config)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0].pixel_roi
    assert 85 <= x1 < x2 <= 115 and 80 <= y1 < y2 <= 120
    assert report["primary"] == "RGB"
    assert boxes[0].height_m == pytest.approx(0.1, abs=0.003)
    # Brief depth holes retain geometry, without inventing a new measurement.
    boxes, report = detector.detect(rgb, depth * 0, intrinsics, config)
    assert len(boxes) == 1 and report["outlines"][0]["geometry"] == "cached"
    detector.previous = [(0, boxes[0])]
    assert detector.detect(rgb, depth * 0, intrinsics, config)[0] == []


def test_rgb_obstacle_geometry_is_clipped_to_operator_workspace():
    config, intrinsics, depth = depth_fixture()
    config["workspace_max_m"][0] = 0.12
    mask = np.zeros_like(depth)
    mask[80:120, 100:130] = 1
    depth[mask.astype(bool)] = 0.5
    boxes, _ = YoloDetector(fake_model(mask)).detect(
        np.zeros((200, 200, 3), dtype=np.uint8),
        depth,
        intrinsics,
        config,
    )
    assert boxes[0].high[0] <= 0.120001


def test_whole_scene_detection_is_not_a_box():
    config, intrinsics, depth = depth_fixture()
    detector = YoloDetector(fake_model(np.ones_like(depth), (0, 0, 200, 200)))
    assert detector.detect(np.zeros((200, 200, 3)), depth, intrinsics, config)[0] == []


def test_paper_instance_with_small_raised_patch_is_still_ignored():
    config, intrinsics, depth = depth_fixture()
    mask = np.zeros_like(depth)
    mask[40:160, 40:160] = 1
    depth[80:90, 80:90] = 0.5
    detector = YoloDetector(fake_model(mask, (40, 40, 160, 160)))
    assert detector.detect(np.zeros((200, 200, 3)), depth, intrinsics, config)[0] == []


def test_different_rgb_rectangles_for_same_raised_geometry_are_deduplicated():
    config, intrinsics, depth = depth_fixture()
    mask = np.zeros_like(depth)
    mask[80:120, 85:115] = 1
    depth[mask.astype(bool)] = 0.5
    first = fake_model(mask).predict()[0]
    second = fake_model(mask, (40, 40, 160, 160)).predict()[0]
    first.boxes.extend(second.boxes)
    torch = pytest.importorskip("torch")
    first.masks.data = torch.cat([first.masks.data, second.masks.data])
    model = SimpleNamespace(predict=lambda *a, **kw: [first])
    boxes, _ = YoloDetector(model).detect(
        np.zeros((200, 200, 3), np.uint8), depth, intrinsics, config
    )
    assert len(boxes) == 1


def test_flat_key_targets_are_kept_separate_and_temporally_confirmed():
    from deskbuddy.trajectory.yolo_perception import (
        mask_grasp_pixel,
        roi_contains_key,
    )

    config, intrinsics, depth = depth_fixture()
    mask = np.zeros_like(depth)
    mask[80:120, 85:115] = 1
    model = fake_model(mask)
    result = model.predict()[0]
    result.names = {0: "keys"}
    detector = YoloDetector(model)
    rgb = np.zeros((200, 200, 3), np.uint8)
    boxes, report = detector.detect(rgb, depth, intrinsics, config)
    assert not boxes and report["key_detections"][0]["confirmed_frames"] == 1
    _, report = detector.detect(rgb, depth, intrinsics, config)
    key = report["key_detections"][0]
    assert key["confirmed_frames"] == 2 and key["grasp_pixel"] == mask_grasp_pixel(mask)
    x, y = key["grasp_pixel"]
    assert mask[y, x]
    assert roi_contains_key((86, 81, 110, 115), key["pixel_roi"])
    assert not roi_contains_key((0, 0, 200, 200), key["pixel_roi"])


def test_slow_inference_does_not_age_out_paired_webcam_and_blue_occlusion_is_warning(
    monkeypatch,
):
    config, intrinsics, depth = depth_fixture()
    config["secondary_camera_calibration"] = {
        "primary_pixels": [],
        "secondary_pixels": [],
    }
    mask = np.zeros_like(depth)
    model = fake_model(mask)
    original = model.predict
    now = [100.0]

    def predict(*a, **kw):
        now[0] += 1.2
        return original(*a, **kw)

    model.predict = predict
    monkeypatch.setattr(
        "deskbuddy.trajectory.yolo_perception.time.time", lambda: now[0]
    )
    monkeypatch.setattr("deskbuddy.multiview.calibration_current", lambda *a: True)
    monkeypatch.setattr("deskbuddy.multiview.blue_agreement", lambda *a: 0.05)
    monkeypatch.setattr(
        "deskbuddy.trajectory.multiview_evidence.secondary_instances", lambda *a: []
    )
    monkeypatch.setattr(
        "deskbuddy.trajectory.multiview_evidence.filter_secondary",
        lambda *a: ([], None),
    )
    detector = YoloDetector(model)
    image = np.zeros((200, 200, 3), np.uint8)
    for _ in range(3):
        _, report = detector.detect(
            image, depth, intrinsics, config, secondary=(image, now[0])
        )
        assert report["secondary_view"]["calibrated"]
    assert "alignment_warning" in report["secondary_view"]
    assert not report["secondary_view"].get("alignment_stale")
