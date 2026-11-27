import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from deskbuddy.config import load_config
from deskbuddy.homing import plan_home
from deskbuddy.kinematics import Arm


def test_home_uses_saved_config_angles_after_reopening(tmp_path):
    config = load_config()
    taught = [62.0, -35.0, 43.0, 54.0, -0.3]
    config["home_pose_deg"] = taught
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    arm = Arm()
    with (
        patch("deskbuddy.trajectory.joint_planner.start_overlap_allowance"),
        patch("deskbuddy.homing.check_self_collision"),
        patch(
            "deskbuddy.homing.plan_joint_route",
            return_value=SimpleNamespace(planner_info=[]),
        ) as route,
    ):
        plan, _ = plan_home(arm, np.zeros(5), loaded, [], 0)
    np.testing.assert_allclose(route.call_args.kwargs["goal_pose"], np.deg2rad(taught))
    assert plan.planner_info[-1]["home_source"] == "saved pose"


def test_saved_home_is_not_silently_clipped_to_other_pose():
    config = dict(load_config(), home_pose_deg=[999, 0, 0, 0, 0])
    with patch("deskbuddy.trajectory.joint_planner.start_overlap_allowance"):
        with pytest.raises(ValueError, match="Saved Home outside model limits"):
            plan_home(Arm(), np.zeros(5), config, [], 0)
