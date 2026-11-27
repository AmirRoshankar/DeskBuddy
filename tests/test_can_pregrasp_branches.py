import numpy as np

from deskbuddy.can_discard import center_ik, jaw_center, opening_for_width
from deskbuddy.grasp_geometry import tool_geometry
from deskbuddy.kinematics import Arm


def test_can_pregrasp_from_far_wrist_roll_uses_valid_downward_solution():
    arm = Arm()
    seed = np.array([0.02915, -1.4008, 1.2134, 0.786, -2.801])
    seed = np.clip(seed, arm.lower, arm.upper)
    target = np.array([0.28126, 0.04019, 0.17822])
    percent = opening_for_width(arm, 0.072)
    q = center_ik(arm, target, seed, percent, max_tilt_deg=25)
    assert np.linalg.norm(jaw_center(arm, q, percent) - target) < 0.002
    assert tool_geometry(arm, q)[2] @ [0, 0, -1] > np.cos(np.deg2rad(25))
    assert np.all(q >= arm.lower) and np.all(q <= arm.upper)
