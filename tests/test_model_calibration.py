import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deskbuddy.kinematics import Arm
from deskbuddy.model_calibration import fit_candidate, validate_candidate


def test_corrected_model_validates_only_on_fresh_unmodified_data():
    arm = Arm()
    rng = np.random.default_rng(31)
    correction = np.deg2rad([0, -3, -4, -6, 0])
    r = Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_matrix()
    translation = np.array([0.1, 0.05, -0.1])
    poses = rng.uniform(-0.7, 0.7, (10, 5))
    rows = []
    for q in poses:
        camera = r.T @ (arm.fk(q + correction) - translation)
        rows.append(
            {
                "q_model_rad": q.tolist(),
                "camera_point_m": camera.tolist(),
                "camera_serial": "one",
                "session_id": "same",
                "unix_time_s": 0,
            }
        )
    candidate = fit_candidate(rows[:8])
    assert np.allclose(
        candidate["joint_offset_correction_deg"], [0, -3, -4, -6, 0], atol=0.01
    )
    with pytest.raises(ValueError, match="NEW"):
        validate_candidate(candidate, rows[:8])
    with pytest.raises(ValueError, match="fresh"):
        validate_candidate(candidate, rows)
    for row in rows[8:]:
        row["unix_time_s"] = candidate["created_unix_s"] + 1
    report = validate_candidate(candidate, rows)
    assert report["passed"] and not report["refitted_on_checks"]
    rows[-1]["camera_point_m"][0] += 0.05
    assert not validate_candidate(candidate, rows)["passed"]
