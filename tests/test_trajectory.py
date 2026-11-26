import json

import numpy as np

from deskbuddy.trajectory.collision import Scene
from deskbuddy.trajectory.rrt import resample, rrt_connect, shortcut


def test_segment_crossings_and_tangency_are_rejected_even_with_free_endpoints():
    scene = Scene(circles=((0.1, 0.1, 0.02),), tool_radius=0.01, margin=0)
    a, b = [0.03, 0.1], [0.18, 0.1]
    assert scene.segment_free(a, a) and scene.segment_free(b, b)
    assert not scene.segment_free(a, b)
    assert not scene.segment_free([0.03, 0.13], [0.18, 0.13])
    box = Scene(boxes=((0.08, 0.08, 0.12, 0.12),), tool_radius=0.01, margin=0)
    assert not box.segment_free(a, b)
    assert box.segment_free([0.02, 0.15], [0.18, 0.15])
    assert not box.segment_free([-0.01, 0.15], [0.18, 0.15])


def test_rrt_and_smoothing_preserve_collision_free_endpoints_and_seed():
    scene = Scene(circles=((0.105, 0.1485, 0.035),))
    start, goal = [0.025, 0.1485], [0.185, 0.1485]
    path = rrt_connect(scene, start, goal, np.random.default_rng(123))
    assert path is not None and scene.path_free(path)
    assert np.array_equal(
        path, rrt_connect(scene, start, goal, np.random.default_rng(123))
    )
    smooth = shortcut(scene, path, np.random.default_rng(456))
    sampled = resample(smooth, 32)
    assert sampled.shape == (32, 2) and scene.path_free(sampled)
    assert np.allclose(sampled[0], start) and np.allclose(sampled[-1], goal)
    assert scene.clearance(sampled) > 0
    assert rrt_connect(scene, start, [0.105, 0.1485], np.random.default_rng(1)) is None


def test_resampling_cannot_cut_a_collision_free_corner():
    scene = Scene(circles=((0.085, 0.108, 0.006),), tool_radius=0, margin=0)
    path = np.array([[0.02, 0.1], [0.1, 0.1], [0.1, 0.2]])
    assert scene.path_free(path)
    # Naive arc-uniform resampling connects across the bend and hits the circle.
    naive = np.array([[0.02, 0.1], [0.1, 0.11], [0.1, 0.2]])
    assert not scene.path_free(naive)
    assert scene.path_free(resample(path, 32))


def test_failed_teacher_attempts_are_retained_and_not_training_rows(
    tmp_path, monkeypatch
):
    from deskbuddy.trajectory import dataset

    monkeypatch.setattr(dataset, "rrt_connect", lambda *args: None)
    report = dataset.generate(tmp_path, scenarios=3, routes=2)
    assert len(report["attempts"]) == 6
    assert all(a["status"] == "rrt_failed" for a in report["attempts"])
    assert report["metrics"]["success_rate"] == 0
    assert np.load(tmp_path / "dataset.npz")["paths"].shape == (0, 32, 2)
    assert len(json.loads((tmp_path / "dataset.json").read_text())["scenarios"]) == 3
