from copy import deepcopy

import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.kinematics import Arm
from deskbuddy.table_collision import (
    bodies,
    clearances,
    minimum_clearance,
    path_free,
    physical_plane,
)


def test_body_support_matches_explicit_transformed_mesh_vertices():
    from deskbuddy.table_collision import transformed_bodies

    cfg = deepcopy(DEFAULT)
    arm = Arm()
    q = np.deg2rad([12, -15, 30, 15, 25])
    n, d = physical_plane(cfg)
    expected = {}
    for name, vertices, t in transformed_bodies(arm, q):
        world = vertices @ t[:3, :3].T + t[:3, 3]
        value = (world @ n + d).min() - 0.001 - (0.0001 if name == "moving_jaw" else 0)
        expected[name] = min(value, expected.get(name, np.inf))
    actual = dict(clearances(arm, q, cfg))
    assert all(abs(actual[k] - v) < 1e-10 for k, v in expected.items())
    assert "base_link" not in actual


def test_table_uncertainty_margin_is_not_added_to_physical_body_collision():
    cfg = deepcopy(DEFAULT)
    cfg["test_bed"] = dict(
        corners_base_m=[
            [0.1, -0.1, 0],
            [0.31, -0.1, 0],
            [0.31, 0.197, 0],
            [0.1, 0.197, 0],
        ],
        clearance_margin_m=0.003,
    )
    q = np.deg2rad(DEFAULT["preview_start_deg"])
    first = minimum_clearance(Arm(), q, cfg)
    cfg["test_bed"]["clearance_margin_m"] = 0.04
    assert minimum_clearance(Arm(), q, cfg) == first
    assert first > 0.05


def test_body_data_has_real_gripper_and_jaw_and_continuous_table_validator():
    geometry, _, _ = bodies()
    assert len(geometry["moving_jaw_so101_v1_link"]) > 10
    arm = Arm()
    q = np.deg2rad(DEFAULT["preview_start_deg"])
    assert path_free(arm, [q, q + np.deg2rad([1, 1, 1, 1, 1])], DEFAULT)
    cfg = {**DEFAULT, "table_z_m": 0.20}
    assert not path_free(arm, [q, q], cfg)


def test_cached_jaw_envelope_preserves_support_for_tilted_planes_and_poses():
    from deskbuddy.table_collision import transformed_bodies

    rng = np.random.default_rng(123)
    arm = Arm()
    cfg = deepcopy(DEFAULT)
    cfg["test_bed"] = dict(
        corners_base_m=[
            [0, 0, 0],
            [0.21, 0, 0.01],
            [0.21, 0.297, 0.025],
            [0, 0.297, 0.015],
        ]
    )
    for _ in range(12):
        q = rng.uniform(arm.lower, arm.upper)
        n, d = physical_plane(cfg)
        expected = {}
        for name, vertices, t in transformed_bodies(arm, q):
            value = (
                float((vertices @ (t[:3, :3].T @ n)).min() + n @ t[:3, 3] + d)
                - 0.001
                - (0.0001 if name == "moving_jaw" else 0)
            )
            expected[name] = min(value, expected.get(name, np.inf))
        actual = dict(clearances(arm, q, cfg))
        for name, value in expected.items():
            assert abs(actual[name] - value) < 1e-10
