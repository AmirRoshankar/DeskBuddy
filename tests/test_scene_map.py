from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.scene_map import SceneMap

INTR = {"fx": 100, "fy": 100, "ppx": 10, "ppy": 10}


def test_missing_depth_preserves_observed_cloud_and_marks_age():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = np.eye(4).tolist()
    scene = SceneMap()
    scene.update(np.full((20, 20), 0.6), now=10)
    scene.update(np.zeros((20, 20)), now=20)
    p, source = scene.point(10, 10, INTR, c, now=20)
    assert np.allclose(p, [0, 0, 0.6])
    assert source["source"] == "cached depth" and source["age_s"] == 10
    assert len(scene.point_cloud(INTR, np.eye(4), stride=1)) == 400
    changed = np.full((20, 20), 0.4)
    scene.update(changed, now=21)
    assert scene.point(10, 10, INTR, c, now=21)[0][2] == pytest.approx(0.6)
    scene.update(changed, now=22)
    scene.update(changed, now=23)
    assert scene.point(10, 10, INTR, c, now=23)[0][2] == pytest.approx(0.4)


def test_a4_plane_needs_no_current_or_previous_depth():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = np.eye(4).tolist()
    c["test_bed"] = {
        "corners_base_m": [
            [-0.1, -0.1, 0.6],
            [0.11, -0.1, 0.6],
            [0.11, 0.197, 0.6],
            [-0.1, 0.197, 0.6],
        ],
        "inset_m": 0.02,
    }
    p, source = SceneMap().point(10, 10, INTR, c)
    assert np.allclose(p, [0, 0, 0.6]) and source["source"] == "A4 plane estimate"


def test_depth_discontinuity_and_single_valid_neighbor_do_not_fail():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = np.eye(4).tolist()
    depth = np.full((20, 20), np.nan)
    depth[10, 9] = 0.4
    depth[10, 11] = 0.8
    scene = SceneMap()
    scene.update(depth, now=1)
    p, source = scene.point(10, 10, INTR, c, now=1)
    assert p[2] in (0.4, 0.8) and source["neighbor_pixels"] == 1
    with pytest.raises(ValueError, match="not been observed"):
        scene.point(19, 0, INTR, c)


def test_depth_kalman_suppresses_noise_rejects_spikes_and_grows_uncertainty_without_data():
    rng = np.random.default_rng(5)
    scene = SceneMap()
    raw = []
    filtered = []
    for index in range(300):
        observation = 0.6 + rng.normal(0, 0.0034, (8, 8))
        scene.update(observation, now=index / 30)
        raw.append(observation[4, 4])
        filtered.append(scene.depth[4, 4])
    assert np.std(filtered[80:]) < np.std(raw[80:]) * 0.6
    previous = scene.depth.copy()
    variance = scene.variance.copy()
    scene.update(np.full((8, 8), 1.2), now=10.0)
    assert np.allclose(scene.depth, previous)
    scene.update(np.zeros((8, 8)), now=12.0)
    assert (scene.variance > variance).all()
