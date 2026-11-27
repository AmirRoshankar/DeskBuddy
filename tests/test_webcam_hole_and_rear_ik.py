import cv2
import numpy as np
import pytest

from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.can_opening import dark_spot_center


def test_hole_centroid_ignores_dark_objects_outside_can_and_below_top():
    image = np.full((300, 300, 3), 200, np.uint8)
    cv2.ellipse(image, (150, 80), (12, 7), 0, 0, 360, (15, 15, 15), -1)
    cv2.circle(image, (30, 30), 20, (0, 0, 0), -1)
    cv2.circle(image, (150, 200), 20, (0, 0, 0), -1)
    result = dark_spot_center(image, [100, 50, 200, 250])
    assert np.linalg.norm(np.asarray(result["pixel"]) - [150, 80]) < 1
    assert result["search_roi_px"] == [100, 50, 200, 140]


def test_no_spot_does_not_guess_center():
    with pytest.raises(ValueError, match="No distinct dark spot"):
        dark_spot_center(np.full((300, 300, 3), 200, np.uint8), [100, 50, 200, 250])


def test_behind_base_target_has_ik_within_existing_limits():
    arm = Arm()
    rear_pose = np.deg2rad([0, -60, -90, -85, 0])
    target = arm.fk(rear_pose)
    assert target[0] < 0
    solution = arm.ik(
        target, np.deg2rad([0, -15, 30, 15, 0]), locked_joints=["wrist_roll"]
    )
    assert np.linalg.norm(arm.fk(solution) - target) <= 0.002
    assert np.all(solution >= arm.lower)
    assert np.all(solution <= arm.upper)
