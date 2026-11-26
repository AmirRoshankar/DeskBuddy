import numpy as np
import pytest

from deskbuddy.kinematics import Arm
from deskbuddy.self_collision import check, collisions
from deskbuddy.trajectory.joint_planner import segment_distance


def test_real_body_model_accepts_pose_rejected_by_old_40mm_center_rule():
    arm = Arm([[-110] * 4 + [-180], [110] * 4 + [180]])
    q = np.deg2rad([-34.0616, -106.7681, 86.5425, 64.8382, -24.0816])
    p = arm.frames(q)[0]
    assert segment_distance(p[1], p[2], p[4], p[5]) < 0.04
    assert not collisions(arm, q)
    check(arm, q)


def test_actual_nonadjacent_body_overlap_still_rejected_with_named_links():
    arm = Arm()
    q = np.deg2rad([17.0353, 72.2253, 87.2646, 86.2353, -169.9727])
    with pytest.raises(ValueError, match="URDF body overlap"):
        check(arm, q)
    assert collisions(arm, q)
