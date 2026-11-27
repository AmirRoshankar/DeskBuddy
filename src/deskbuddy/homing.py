"""Plan a rest pose from measured joints; hardware runs only on explicit UI action."""

import copy
import time

import numpy as np

from .gripper_control import ramp_gripper
from .planning import check_workspace
from .table_collision import minimum_clearance
from .trajectory.joint_planner import check_self_collision, plan_joint_route


def plan_home(arm, start, config, boxes, revision):
    cfg = copy.deepcopy(config)
    cfg.update(
        locked_joints=[],
        grasp_request=None,
        home_request=True,
        allow_object_collisions=True,
        workspace_min_m=[-1.0, -1.0, -1.0],
        workspace_max_m=[1.0, 1.0, 1.0],
        restrict_to_test_bed=False,
    )
    try:
        from .trajectory.joint_planner import start_overlap_allowance

        start_overlap_allowance(arm, start)
    except ValueError as exc:
        raise ValueError(f"Home start pose: {exc}") from exc
    saved = config.get("home_pose_deg")
    if saved is None:
        neutral = np.clip(np.deg2rad([0, -15, 30, 15, 0]), arm.lower, arm.upper)
    else:
        neutral = np.deg2rad(np.asarray(saved, float))
        if neutral.shape != (5,) or not np.isfinite(neutral).all():
            raise ValueError(
                "Saved Home must contain five finite joint angles; use Set home again"
            )
        if (neutral < arm.lower).any() or (neutral > arm.upper).any():
            raise ValueError(arm.limit_error(neutral, "Saved Home"))
    goals = [neutral]
    # Home is recovery to a fixed rest pose, independent of the selection box.
    # Only this command's private configuration has the recovery workspace.
    reasons = []
    for goal in goals:
        try:
            target = arm.fk(goal)
            check_workspace(target, cfg)
            check_self_collision(arm, goal)
            if (
                not cfg["allow_object_collisions"]
                and minimum_clearance(arm, goal, cfg) < 0
            ):
                raise ValueError("Rest pose body touches table")
            plan = plan_joint_route(
                arm, start, target, cfg, boxes, 1, revision, goal_pose=goal
            )
            plan.planner_info.append(
                dict(
                    home_preview=True,
                    home_pose_deg=np.rad2deg(goal).tolist(),
                    home_source="saved pose" if saved is not None else "default pose",
                )
            )
            return plan, cfg
        except ValueError as exc:
            reasons.append(str(exc))
    raise ValueError("Home unavailable: " + "; ".join(dict.fromkeys(reasons)))


def execute_home(
    bus, selected, config, stopped, publish, sleep=time.sleep, read_scene=None
):
    from .execute_reach import execute_target

    # execute_target revalidates the saved route and checks fresh encoder hold.
    plan, result = execute_target(
        bus,
        selected.target_m,
        config,
        stopped,
        publish,
        selected_plan=selected,
        sleep=sleep,
        read_scene=read_scene,
    )
    if stopped():
        raise RuntimeError("Home stopped; hold retained")
    grip = ramp_gripper(bus, 0.0, stopped, sleep)
    result.update(grip, home_completed=True, gripper_unchanged=False)
    return plan, result
