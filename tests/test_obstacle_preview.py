from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.trajectory.collision import Scene
from deskbuddy.trajectory.perception import detect_boxes


def depth_fixture():
    config = deepcopy(DEFAULT)
    config.update(
        calibration_verified=True,
        workspace_min_m=[0, 0, 0],
        workspace_max_m=[0.21, 0.297, 0.4],
        T_base_camera=[
            [1, 0, 0, 0.105],
            [0, -1, 0, 0.1485],
            [0, 0, -1, 0.6],
            [0, 0, 0, 1],
        ],
    )
    intrinsics = dict(fx=200.0, fy=200.0, ppx=100.0, ppy=100.0)
    return config, intrinsics, np.full((200, 200), 0.6)


def test_depth_box_estimates_and_swaps_without_manual_dimensions():
    config, intrinsics, depth = depth_fixture()
    assert detect_boxes(depth, intrinsics, config, stride=1, min_pixels=25)[0] == []
    depth[80:120, 85:115] = 0.5
    boxes, _ = detect_boxes(depth, intrinsics, config, stride=1, min_pixels=25)
    assert len(boxes) == 1
    assert np.allclose(boxes[0].high - boxes[0].low, [0.075, 0.10, 0.1], atol=0.006)
    depth[:] = 0.6
    depth[95:105, 95:105] = 0.55
    swapped, _ = detect_boxes(depth, intrinsics, config, stride=1, min_pixels=25)
    assert len(swapped) == 1 and swapped[0].height_m == pytest.approx(0.05, abs=0.002)
    assert (swapped[0].high - swapped[0].low < boxes[0].high - boxes[0].low).all()


def test_invalid_depth_does_not_create_an_obstacle():
    config, intrinsics, depth = depth_fixture()
    depth[80:120, 85:115] = 0
    depth[20:30, 20:30] = np.nan
    assert detect_boxes(depth, intrinsics, config, stride=1, min_pixels=25)[0] == []


def test_static_path_direction_is_restored_after_canonicalization(monkeypatch):
    pytest.importorskip("torch")
    from deskbuddy.trajectory import live_preview

    suite = live_preview.PlannerSuite.__new__(live_preview.PlannerSuite)
    suite.models = {name: object() for name in ("mlp", "fm", "mf", "imf")}
    norm = dict(c_mean=[0] * 18, c_std=[1] * 18, y_mean=[0] * 60, y_std=[1] * 60)
    suite.checkpoints = {name: dict(normalization=norm) for name in suite.models}
    monkeypatch.setattr(
        live_preview,
        "sample",
        lambda model, c, noise, method, steps: c.new_zeros((len(c), 60)),
    )
    scene = Scene()
    start, goal = [0.18, 0.2], [0.02, 0.1]
    results = suite.compare(scene, start, goal, candidates=4)
    for row in results:
        assert np.allclose(row["selected_path"][0], start)
        assert np.allclose(row["selected_path"][-1], goal)
        assert row["valid_candidates"] == row["candidates"] and not row["fallback"]
    assert all(row["canonical_direction_reversed"] for row in results[:-1])


def test_grouping_extra_objects_contains_every_original_footprint():
    from deskbuddy.trajectory.perception import DetectedBox, grouped_footprints

    boxes = [
        DetectedBox(
            np.array([x, y, 0]),
            np.array([x + 0.02, y + 0.02, 0.05]),
            1000,
            0.05,
            (0, 0, 10, 10),
        )
        for x, y in ((0.02, 0.02), (0.05, 0.02), (0.12, 0.13), (0.15, 0.13))
    ]
    grouped = grouped_footprints(boxes)
    assert len(grouped) == 2
    for box in boxes:
        for point in (box.low[:2], box.high[:2]):
            assert any(((point >= r[:2]) & (point <= r[2:])).all() for r in grouped)


def test_selected_planner_skips_unrequested_models_and_rrt(monkeypatch):
    from deskbuddy.trajectory import live_preview

    suite = live_preview.PlannerSuite.__new__(live_preview.PlannerSuite)
    suite.models = {"imf": object()}
    suite.checkpoints = {
        "imf": dict(
            normalization=dict(
                c_mean=[0] * 18, c_std=[1] * 18, y_mean=[0] * 60, y_std=[1] * 60
            )
        )
    }
    calls = []

    def sample(model, context, noise, method, steps):
        calls.append(method)
        return context.new_zeros((len(context), 60))

    def forbidden(*args):
        raise AssertionError("RRT must not run for an iMF-only request")

    monkeypatch.setattr(live_preview, "sample", sample)
    monkeypatch.setattr(live_preview, "classical", forbidden)
    results = suite.compare(Scene(), [0.02, 0.1], [0.18, 0.2], methods=("imf",))
    assert calls == ["imf"]
    assert [row["method"] for row in results] == ["imf"]
