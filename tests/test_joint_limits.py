import numpy as np
import pytest

from deskbuddy.joint_limits import calibrated_limits
from deskbuddy.kinematics import JOINTS, Arm


def test_saved_encoder_ranges_match_degree_normalization_with_offsets_and_signs():
    calibration = {name: dict(range_min=800, range_max=3200) for name in JOINTS}
    signs, offsets = [1, -1, 1, -1, 1], [0, -3, -6, -5, 0]
    bounds = np.array(calibrated_limits(calibration, signs, offsets))
    half = 1200 * 360 / 4095
    assert np.allclose(bounds[0], np.array(offsets) - half)
    assert np.allclose(bounds[1], np.array(offsets) + half)


def test_explicit_calibrated_ranges_replace_generic_urdf_clamp():
    bounds = [[-104, -108.48, -104.89, -108.37, -180], [104, 102.24, 91.94, 97.78, 180]]
    arm = Arm(bounds)
    pose = np.deg2rad([3.34, -107.16, 91.06, 21.48, 3.12])
    assert (pose >= arm.lower).all() and (pose <= arm.upper).all()
    assert np.rad2deg(arm.lower[1]) == pytest.approx(-108.48)
    assert np.rad2deg(Arm().lower[1]) == pytest.approx(-100, abs=0.001)
    # A target at an already valid measured pose must be a valid IK seed.
    assert np.linalg.norm(arm.fk(arm.ik(arm.fk(pose), pose)) - arm.fk(pose)) < 0.002


def test_limit_diagnostic_names_joint_actual_angle_and_source():
    arm = Arm()
    message = arm.limit_error(np.deg2rad([0, -107.2, 0, 0, 0]))
    assert "shoulder_lift -107.20°" in message
    assert "allowed -100.00" in message and "URDF defaults" in message
    with pytest.raises(ValueError, match="lower limits"):
        Arm([[10] * 5, [0] * 5])
