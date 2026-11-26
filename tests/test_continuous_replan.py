import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.execute_reach import (
    MotionStopped,
    SceneReplanRequested,
    execute_target,
)
from deskbuddy.kinematics import JOINTS
from deskbuddy.planning import Plan


def test_replan_resumes_from_measured_pose_without_new_go(monkeypatch):
    config = deepcopy(DEFAULT)
    config["joint_signs"] = [1] * 5
    config["joint_offsets_deg"] = [0] * 5
    current = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    calls = []
    planned = []
    initial = Plan(
        np.array([0.0, 1.0]),
        np.zeros((2, 5)),
        np.array([0.2, 0, 0.1]),
        0.0,
        planner_info=[dict(trajectory_method="rrt")],
    )
    replacement = Plan(
        np.array([0.0, 1.0]),
        np.vstack([current, current + 0.01]),
        initial.target_m,
        0.0,
    )

    def once(*args, **kwargs):
        calls.append(kwargs["selected_plan"])
        if len(calls) == 1:
            raise SceneReplanRequested("Object moved")
        return kwargs["selected_plan"], dict(completed=True)

    def plan(arm, start, *args):
        planned.append(start.copy())
        return replacement

    monkeypatch.setattr("deskbuddy.execute_reach._execute_target_once", once)
    monkeypatch.setattr(
        "deskbuddy.trajectory.execution.RoutePlanner",
        lambda: SimpleNamespace(plan=plan),
    )
    bus = SimpleNamespace(
        sync_read=lambda *a, **k: dict(zip(JOINTS, np.rad2deg(current)))
    )
    output, result = execute_target(
        bus,
        initial.target_m,
        config,
        lambda: False,
        selected_plan=initial,
        read_scene=lambda: dict(
            unix_time_s=time.time(), obstacle_boxes=[], obstacle_detection={}
        ),
    )
    assert output is replacement and result["scene_replans"] == 1
    assert np.allclose(planned[0], current)
    assert len(calls) == 2


def test_operator_stop_cannot_be_auto_resumed(monkeypatch):
    monkeypatch.setattr(
        "deskbuddy.execute_reach._execute_target_once",
        lambda *a, **k: pytest.fail("Must not execute"),
    )
    with pytest.raises(MotionStopped, match="Operator"):
        execute_target(None, np.zeros(3), DEFAULT, lambda: True)
