import numpy as np
import pytest

from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.table import conservative_height, estimate_table


def test_table_plane_recovers_height_with_noise_and_foreground():
    rng = np.random.default_rng(14)
    depth = 0.65 + rng.normal(0, 0.0006, (480, 640))
    depth[290:360, 260:370] = 0.60
    p = estimate_table(
        depth, {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240}, T_BASE_CAMERA
    )
    assert abs(p["height_at_base_origin_m"]) < 0.001
    assert p["tilt_deg"] < 0.3
    assert p["rmse_mm"] < 1.5
    height = conservative_height(p, [0.1, -0.2, 0], [0.4, 0.2, 0.3])
    assert 0 < height < 0.006


def test_table_rejects_missing_depth():
    with pytest.raises(ValueError, match="Not enough"):
        estimate_table(
            np.zeros((480, 640)),
            {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240},
            T_BASE_CAMERA,
        )
