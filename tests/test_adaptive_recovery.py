from copy import deepcopy

import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.kinematics import Arm
from deskbuddy.planning import plan_reach


def test_small_tracking_deflection_can_recover_upward_without_worsening_clearance():
    config = deepcopy(DEFAULT)
    config["workspace_min_m"][2] = 0.015
    arm = Arm()
    seed = np.deg2rad(config["preview_start_deg"])
    start = arm.ik([0.30, 0, 0.014], seed, 0.0001)
    plan = plan_reach(arm, start, [0.29, 0.01, 0.06], config)
    assert plan.recovery_waypoint_m is not None
    deficit = np.array(
        [np.maximum(0.015 - arm.frames(q)[0][2:, 2], 0.0) for q in plan.joints_rad]
    )
    assert np.diff(deficit, axis=0).max() <= 0.00005
    assert deficit[-1].max() == 0.0
