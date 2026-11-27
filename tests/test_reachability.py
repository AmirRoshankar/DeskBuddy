import numpy as np
import pytest

from deskbuddy import reachability
from deskbuddy.kinematics import Arm


@pytest.fixture(scope="module")
def reach():
    arm = Arm()
    # A coarse grid keeps the test fast; the guarantees hold at any step.
    return arm, reachability.ReachMap(arm, 0.0, step_deg=3.0, cell_m=0.004)


def test_no_reachable_configuration_is_ever_called_unreachable(reach):
    arm, rmap = reach
    q = np.random.default_rng(3).uniform(arm.lower, arm.upper, (3000, 5))
    q[:, 4] = rmap.roll
    tips = reachability._batched_tip(arm, q)
    assert np.allclose(tips[0], arm.fk(q[0]))
    assert not any(rmap.outside(p) for p in tips)


def test_reachable_classification_is_backed_by_an_ik_solution(reach):
    arm, rmap = reach
    rng = np.random.default_rng(4)
    q = rng.uniform(arm.lower, arm.upper, (40, 5))
    q[:, 4] = rmap.roll
    for p in reachability._batched_tip(arm, q):
        solution = rmap.certify(p)
        if solution is not None:
            assert np.linalg.norm(arm.fk(solution) - p) <= 0.002
            assert (solution >= arm.lower).all() and (solution <= arm.upper).all()


def test_far_points_are_proven_unreachable_with_a_distance(reach):
    arm, rmap = reach
    assert rmap.outside([1.0, 0.0, 0.1])
    assert rmap.distance_outside([1.0, 0.0, 0.1]) > 0.4
    low, high = rmap.bounds(0.0)
    assert high[0] < 0.6 and low[2] == 0.0
