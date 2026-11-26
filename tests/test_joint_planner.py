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
