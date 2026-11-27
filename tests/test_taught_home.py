from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.homing import plan_home
from deskbuddy.kinematics import Arm


def test_untaught_home_keeps_default_recovery_separate_from_saved_home():
    arm = Arm()
    config = dict(DEFAULT)
    with (
        patch("deskbuddy.trajectory.joint_planner.start_overlap_allowance"),
        patch("deskbuddy.homing.check_self_collision"),
        patch(
            "deskbuddy.homing.plan_joint_route",
            return_value=SimpleNamespace(planner_info=[]),
        ),
    ):
        plan, _ = plan_home(arm, np.zeros(5), config, [], 0)
    assert config["home_pose_deg"] is None
    assert plan.planner_info[-1]["home_source"] == "default pose"


def test_home_preserves_taught_angles_without_sign_changes():
    arm = Arm()
    goal = [0, -40, 70, 10, 0]
    cfg = dict(DEFAULT, home_pose_deg=goal)
    with (
        patch("deskbuddy.trajectory.joint_planner.start_overlap_allowance"),
        patch("deskbuddy.homing.check_self_collision"),
        patch(
            "deskbuddy.homing.plan_joint_route",
            return_value=SimpleNamespace(planner_info=[]),
        ) as route,
    ):
        plan, private = plan_home(arm, np.zeros(5), cfg, [], 0)
    assert np.allclose(route.call_args.kwargs["goal_pose"], np.deg2rad(goal))
    assert plan.planner_info[-1]["home_pose_deg"] == pytest.approx(goal)
