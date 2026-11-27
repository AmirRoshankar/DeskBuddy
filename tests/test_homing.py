import numpy as np
import pytest
from test_execute_reach import setup

from deskbuddy import execute_reach
from deskbuddy.homing import execute_home, plan_home
from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.joint_planner import check_self_collision


def test_home_uses_private_recovery_bounds_without_changing_config():
    cfg, _ = setup()
    arm = Arm()
    start = np.deg2rad(cfg["preview_start_deg"]) + np.deg2rad([3, 1, -2, 2, 15])
    plan, job = plan_home(arm, start, cfg, [], 7)
    assert np.allclose(plan.joints_rad[0], start)
    assert np.allclose(plan.joints_rad[-1], np.deg2rad([0, -15, 30, 15, 0]))
    assert job["home_request"] and job["grasp_request"] is None
    assert not cfg["home_request"]
    assert job["workspace_min_m"] == [-1.0, -1.0, -1.0]
    assert cfg["workspace_min_m"] != job["workspace_min_m"]
    for q in plan.joints_rad:
        check_self_collision(arm, q)


@pytest.mark.parametrize("stop_after_reach", [False, True])
def test_home_closes_after_reach_and_honors_stop(monkeypatch, stop_after_reach):
    calls = []
    selected = object()

    def reached(*args, **kwargs):
        calls.append("reach")
        return selected, dict(completed=True)

    def grip(bus, percent, stopped, sleep):
        calls.append(("grip", percent))
        return dict(gripper_closed=True, gripper_percent=0)

    monkeypatch.setattr(execute_reach, "execute_target", reached)
    monkeypatch.setattr("deskbuddy.homing.ramp_gripper", grip)
    from types import SimpleNamespace

    plan = SimpleNamespace(target_m=np.zeros(3))
    if stop_after_reach:
        with pytest.raises(RuntimeError, match="Home stopped"):
            execute_home(None, plan, {}, lambda: bool(calls), lambda _: None)
        assert calls == ["reach"]
    else:
        _, result = execute_home(None, plan, {}, lambda: False, lambda _: None)
        assert (
            calls == ["reach", ("grip", 0)]
            and result["home_completed"]
            and result["gripper_closed"]
        )


def test_home_starts_from_rest_pose_with_tiny_model_self_overlap():
    from deskbuddy.trajectory.joint_planner import self_overlap

    arm = Arm([[-103.7, -108.5, -104.9, -108.4, -180], [103.7, 102.2, 91.9, 97.8, 180]])
    cfg, _ = setup()
    start = np.deg2rad([11, -107.5, 91.0, 8.5, 10.6])
    overlap = self_overlap(arm, start)
    assert 0 < overlap < 0.003
    plan, _ = plan_home(arm, start, cfg, [], 3)
    assert np.allclose(plan.joints_rad[0], start)
    overlaps = [self_overlap(arm, q) for q in plan.joints_rad[:40]]
    assert all(b <= a + 1e-5 for a, b in zip(overlaps, overlaps[1:]) if a > 0)
    for q in plan.joints_rad[-5:]:
        check_self_collision(arm, q)
