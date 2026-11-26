from copy import deepcopy

import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.diagnostics import running_tracking_limits, tracking_fault
from deskbuddy.grid_preview import assess_grid


def test_tracking_fault_names_the_joint_and_clears_when_feedback_recovers():
    actual = np.zeros(5)
    actual[1] = np.deg2rad(10.15)
    motor = {
        "q_model_rad": actual.tolist(),
        "q_goal_model_rad": [0.0] * 5,
        "q_reference_model_rad": [0.0] * 5,
    }
    fault = tracking_fault(motor)
    assert "shoulder lift" in fault and "10.2" in fault
    motor["q_model_rad"][1] = np.deg2rad(3.8)
    assert tracking_fault(motor) is None


def test_running_limits_apply_to_new_motion_error_not_static_deflection():
    limits = running_tracking_limits([0.5, 4.1, 2.8, 0.2, 0])
    assert np.allclose(limits, [10, 10, 10, 10, 10])
    motor = {
        "q_model_rad": np.deg2rad([0, 4.2, 10.1, 0, 0]).tolist(),
        "q_goal_model_rad": [0.0] * 5,
        "q_reference_model_rad": np.deg2rad([0, 4.2, 0, 0, 0]).tolist(),
        "tracking_limits_deg": limits.tolist(),
    }
    assert "elbow flex" in tracking_fault(motor)
    motor.pop("q_reference_model_rad")
    assert tracking_fault(motor) is None


def test_compact_raised_grid_checks_paths_and_visibility_without_motion():
    config = deepcopy(DEFAULT)
    config.update(
        workspace_min_m=[0.24, -0.04, 0.04],
        workspace_max_m=[0.34, 0.08, 0.3],
        hover_m=0.12,
        T_base_camera=T_BASE_CAMERA.tolist(),
    )
    seed = np.deg2rad(config["preview_start_deg"])
    report = assess_grid(config, seed, {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240})
    assert report["valid_points"] == report["total_points"] == 9
    assert not report["physical_execution"]
    assert all(p["fully_visible"] and p["valid"] for p in report["points"])
