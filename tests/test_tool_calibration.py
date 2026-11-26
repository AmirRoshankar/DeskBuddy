import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deskbuddy.tool_calibration import fit_observations


def observations():
    camera = np.random.default_rng(4).uniform(
        [-0.1, -0.1, 0.35], [0.1, 0.1, 0.55], (8, 3)
    )
    r = Rotation.from_euler("xyz", [0.4, 0.2, -0.6]).as_matrix()
    base = camera @ r.T + [0.1, -0.1, 0.05]
    return [
        {
            "camera_point_m": a.tolist(),
            "base_point_model_m": b.tolist(),
            "session_id": "one",
            "camera_serial": "cam",
        }
        for a, b in zip(camera, base)
    ]


def test_recovers_rigid_transform_on_heldout_points():
    report = fit_observations(observations())
    assert report["passed"]
    assert report["fit_points"] == 6
    assert max(report["heldout_error_mm"]) < 1e-9


def test_heldout_error_is_not_hidden_by_refitting():
    rows = observations()
    rows[-1]["base_point_model_m"][0] += 0.05
    report = fit_observations(rows)
    assert not report["passed"]
    assert max(report["heldout_error_mm"]) > 40
    assert report["fit_rmse_mm"] < 1e-9


def test_sessions_and_insufficient_geometry_rejected():
    rows = observations()
    rows[-1]["session_id"] = "two"
    with pytest.raises(ValueError, match="sessions"):
        fit_observations(rows)
    rows = observations()
    for row in rows:
        row["base_point_model_m"][2] = 0.1
    with pytest.raises(ValueError, match="spread"):
        fit_observations(rows)
    with pytest.raises(ValueError, match="eight"):
        fit_observations(rows[:5])
