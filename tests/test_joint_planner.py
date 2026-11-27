from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.trajectory.joint_planner import JointSpace, check_self_collision


def test_joint_space_checks_link_bodies_and_swept_edges_not_only_tip():
    class Arm:
        lower = np.full(5, -1.0)
        upper = np.full(5, 1.0)

        def frames(self, q):
            return np.array(
                [[0, 0, 0.1], [0.2, q[0], 0.1], [0.2, q[0], 0.3], [0.3, q[0], 0.3]]
            ), None

    config = deepcopy(DEFAULT)
    config.update(workspace_min_m=[0, -0.4, 0], workspace_max_m=[0.4, 0.4, 0.4])
    obstacle = dict(low_m=[0.09, -0.01, 0.07], high_m=[0.11, 0.01, 0.13])
    a = np.zeros(5)
    a[0] = -0.2
    b = a.copy()
    b[0] = 0.2
    space = JointSpace(Arm(), config, [obstacle], a)
    assert space.valid(a) and space.valid(b)
    assert not space.valid(np.zeros(5))
    assert not space.segment_free(a, b)


def test_non_neighboring_self_collision_is_rejected():
    class Arm:
        chain = ["test"]

        def frames(self, q):
            return np.array(
                [
                    [-0.1, 0, 0.1],
                    [0.1, 0, 0.1],
                    [0.2, 0.2, 0.1],
                    [-0.2, 0.2, 0.1],
                    [0, -0.1, 0.1],
                ]
            ), None

    with pytest.raises(ValueError, match="Self collision"):
        check_self_collision(Arm(), np.zeros(5))


def _rest_case():
    from deskbuddy.kinematics import Arm

    arm = Arm()
    config = deepcopy(DEFAULT)
    start = np.deg2rad(DEFAULT["preview_start_deg"])
    return arm, config, start


def test_measured_start_just_past_a_joint_stop_recovers_instead_of_failing():
    from deskbuddy.trajectory.joint_planner import plan_joint_route

    arm, config, start = _rest_case()
    start = start.copy()
    start[1] = arm.lower[1] - np.deg2rad(0.6)
    goal = np.deg2rad(DEFAULT["preview_start_deg"]) + np.deg2rad([10, 5, -5, 0, 0])
    plan = plan_joint_route(arm, start, arm.fk(goal), config, [], 1, 0, goal_pose=goal)
    q = np.asarray(plan.joints_rad)
    assert np.allclose(q[0], start)
    excess = np.maximum(arm.lower - q, 0)[:, 1]
    assert (np.diff(excess) <= 1e-9).all() and excess[-1] == 0
    info = plan.planner_info[0]["start_recovery"]
    assert "shoulder_lift" in info["reason"]


def test_start_far_outside_joint_limits_still_reports_the_joint():
    from deskbuddy.trajectory.joint_planner import plan_joint_route

    arm, config, start = _rest_case()
    start = start.copy()
    start[1] = arm.lower[1] - np.deg2rad(5)
    with pytest.raises(ValueError, match="shoulder_lift"):
        plan_joint_route(arm, start, arm.fk(start), config, [], 1, 0)


def test_blocked_target_names_the_blocking_constraint():
    from deskbuddy.trajectory.joint_planner import plan_joint_route

    arm, config, start = _rest_case()
    obstacle = dict(low_m=[0.27, 0.12, 0.0], high_m=[0.33, 0.18, 0.1])
    with pytest.raises(
        ValueError, match="every target posture is blocked: .+IK postures"
    ):
        plan_joint_route(
            arm, start, np.array([0.3, 0.15, 0.08]), config, [_Box(obstacle)], 1, 0
        )


class _Box:
    def __init__(self, geometry):
        self.geometry = geometry

    def as_dict(self):
        return self.geometry
