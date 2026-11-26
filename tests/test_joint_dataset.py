from copy import deepcopy

import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.joint_dataset import (
    METHODS,
    condition,
    decode_path,
    encode_path,
    layout,
    resample_joint,
)


def test_joint_representation_retains_posture_and_six_distinct_3d_obstacles():
    cfg, boxes, normal, offset = layout(deepcopy(DEFAULT), 6, 123)
    arm = Arm()
    start = np.deg2rad(DEFAULT["preview_start_deg"])
    goal = start + np.deg2rad([20, 10, -10, 15, 5])
    case = dict(
        config=cfg,
        boxes=boxes,
        table_normal=normal,
        plane_offset_m=offset,
        start=start,
        goal=goal,
    )
    limits = [arm.lower, arm.upper]
    c = condition(case, limits)
    assert c.shape == (67,) and len(boxes) == 6
    moved = {**case, "start": start + 0.1}
    assert not np.array_equal(condition(moved, limits), c)
    path = resample_joint([start, start + 0.05, goal])
    assert path.shape == (32, 5)
    assert np.allclose(decode_path(encode_path(path, case, limits), case, limits), path)
    assert METHODS == ("mlp", "fm", "imf")


def test_layout_determinism_and_height_diversity_without_scaling_the_robot():
    cfg, boxes, normal, _ = layout(deepcopy(DEFAULT), 6, 123)
    other = layout(deepcopy(DEFAULT), 6, 123)
    assert cfg == other[0] and boxes == other[1]
    assert abs(np.linalg.norm(normal) - 1) < 1e-10
    assert cfg["model_joint_limits_deg"] == DEFAULT["model_joint_limits_deg"]
    assert cfg["tool_offset_m"] == DEFAULT["tool_offset_m"]
    assert len({round(b["height_above_table_m"], 4) for b in boxes}) > 1
