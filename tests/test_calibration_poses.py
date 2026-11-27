from unittest.mock import patch

import numpy as np
import pytest

from deskbuddy.calibration_poses import choose_poses
from deskbuddy.config import load_config
from deskbuddy.kinematics import Arm


def test_blocked_targets_skipped_before_route_collection():
    arm = Arm()
    attempted = []

    class Space:
        def __init__(self, *args):
            pass

        def reason(self, q):
            return "table" if q[1] > 0 else None

    def route(a, b):
        assert b[1] <= 0
        attempted.append((a.copy(), b.copy()))

    with patch("deskbuddy.trajectory.joint_planner.JointSpace", Space):
        poses, report = choose_poses(arm, np.zeros(5), load_config(), route)
    assert len(poses) == 15
    assert report["rejected"] > 0
    assert len(attempted) == 14
    for i, (a, b) in enumerate(attempted):
        assert np.allclose(a, poses[i])
        assert np.allclose(b, poses[i + 1])


def test_insufficient_poses_explains_block_without_motion():
    class Space:
        def __init__(self, *args):
            pass

        def reason(self, q):
            return "wrist table clearance -20 mm"

    with patch("deskbuddy.trajectory.joint_planner.JointSpace", Space):
        with pytest.raises(ValueError, match="Only 1 route-checked poses.*wrist table"):
            choose_poses(
                Arm(),
                np.zeros(5),
                load_config(),
                lambda *args: pytest.fail("Must not plan blocked pose"),
            )
