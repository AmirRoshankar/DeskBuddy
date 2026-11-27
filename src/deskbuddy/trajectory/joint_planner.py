"""Configuration-space RRT-Connect with whole-arm state/edge validation."""

import copy
import json
import threading
import time

import numpy as np

from ..kinematics import JOINTS
from ..planning import Plan, validate_plan
from ..table_collision import minimum_clearance
from .cancel import PlanningCancelled, raise_if_cancelled
from .rrt import rrt_connect, shortcut


def segment_distance(a, b, c, d):
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = u @ u, u @ v, v @ v, u @ w, v @ w
    candidates = []
    for p, x, y in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)):
        delta = y - x
        t = np.clip((p - x) @ delta / max(delta @ delta, 1e-20), 0, 1)
        candidates.append(np.linalg.norm(p - (x + t * delta)))
    determinant = aa * cc - bb * bb
    if determinant > 1e-16:
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w + s * u - t * v))
    return float(min(candidates))


def check_self_collision(arm, q):
    if not hasattr(arm, "chain"):
        return
    if arm.chain and isinstance(arm.chain[0], tuple):
        from ..self_collision import check

        return check(arm, q)
    points = arm.frames(q)[0]
    # Neighboring links share joints; the wrist, fixed jaw and tip are one body.
    for i in range(len(points) - 1):
        for j in range(i + 3, len(points) - 1):
            if i >= 4 and j >= 4:
                continue
            if (
                segment_distance(points[i], points[i + 1], points[j], points[j + 1])
                < 0.04
            ):
                raise ValueError(
                    f"Self collision between arm links {i + 1} and {j + 1}"
                )


# Parked/rest poses can show a sub-millimetre overlap between the convex URDF
# hulls (a model artifact, not contact). A measured start with at most this
# much overlap may move away from it, as long as the overlap never grows.
MAX_START_SELF_OVERLAP_M = 0.003


def self_overlap(arm, q):
    """Largest predicted self-overlap radius in metres (0 when clear)."""
    if not (hasattr(arm, "chain") and arm.chain and isinstance(arm.chain[0], tuple)):
        try:
            check_self_collision(arm, q)
            return 0.0
        except ValueError:
            return np.inf
    from ..self_collision import collisions

    hits = collisions(arm, q)
    return max((h["overlap_radius_m"] for h in hits), default=0.0)


def start_overlap_allowance(arm, q):
    """Overlap a measured start may begin with; raises if it is real contact."""
    overlap = self_overlap(arm, q)
    if overlap > MAX_START_SELF_OVERLAP_M:
        check_self_collision(arm, q)
    return overlap


def check_self_collision_path(arm, joints, step=np.deg2rad(0.5)):
    """Self-collision along a joint path at a fixed 0.5 degree resolution.

    Dense control-rate samples (250 Hz) are far finer than this, so checking
    every sample repeated near-identical work; sparse samples are densified.
    """
    joints = np.asarray(joints, float)
    if len(joints) == 0:
        return
    checked = joints[0]
    # Tiny start overlap: only non-increasing overlap until the path is clear.
    allowance = start_overlap_allowance(arm, checked)

    def check(q):
        nonlocal allowance
        if allowance > 0:
            overlap = self_overlap(arm, q)
            if overlap > allowance + 1e-5:
                check_self_collision(arm, q)
            allowance = overlap
        else:
            check_self_collision(arm, q)

    for a, b in zip(joints[:-1], joints[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for t in np.linspace(0, 1, count + 1)[1:]:
            q = a + (b - a) * t
            if np.max(np.abs(q - checked)) >= step * 0.999:
                raise_if_cancelled()
                check(q)
                checked = q
    if np.max(np.abs(joints[-1] - checked)) > 1e-12:
        check(joints[-1])


class JointSpace:
    def __init__(self, arm, config, obstacles, start, seconds=5):
        self.arm, self.config, self.obstacles = arm, config, obstacles
        self.low, self.high = arm.lower.copy(), arm.upper.copy()
        self.deadline = time.monotonic() + seconds
        self.cache = {}
        self.checks = 0
        for j, name in enumerate(JOINTS):
            if name in config.get("locked_joints", []):
                self.low[j] = self.high[j] = start[j]

    def valid(self, q):
        key = tuple(np.round(q, 7))
        if key in self.cache:
            return self.cache[key]
        valid = self.reason(q) is None
        self.cache[key] = valid
        return valid

    def reason(self, q):
        """Why a posture is invalid (None when valid); used for clear rejections."""
        from .execution import check_pose_obstacles

        self.checks += 1
        try:
            if (q < self.low - 1e-8).any() or (q > self.high + 1e-8).any():
                raise ValueError("joint limit")
            points = self.arm.frames(q)[0]
            if not self.config.get("allow_object_collisions"):
                clearance = minimum_clearance(self.arm, q, self.config)
                if clearance < -1e-8:
                    raise ValueError(
                        f"arm body {-clearance * 1000:.1f} mm inside table clearance"
                    )
            tip = points[-1]
            if (tip < self.config["workspace_min_m"]).any() or (
                tip > self.config["workspace_max_m"]
            ).any():
                raise ValueError("tool outside workspace bounds")
            check_self_collision(self.arm, q)
            if not self.config.get("allow_object_collisions"):
                check_pose_obstacles(self.arm, q, self.obstacles)
        except ValueError as exc:
            return str(exc)
        return None

    def segment_free(self, a, b):
        raise_if_cancelled()
        if time.monotonic() > self.deadline:
            raise ValueError("Whole-arm RRT time budget reached")
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(1))))
        return all(self.valid(a + (b - a) * t) for t in np.linspace(0, 1, count + 1))

    def path_free(self, path):
        return all(self.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))


# Measured starts may sit slightly outside the planning constraints (encoder
# noise at a measured joint stop, a parked arm a few millimetres inside the
# table clearance band). These match the recovery limits validate_plan accepts.
MAX_START_JOINT_EXCESS = np.deg2rad(2.0)
MAX_START_TABLE_DEFICIT_M = 0.005
MAX_START_WORKSPACE_Z_M = 0.015


def start_violation(space, q):
    """Joint excess, table deficit and tip workspace violation; zero when inside."""
    arm, config = space.arm, space.config
    joints = np.maximum(np.maximum(arm.lower - q, q - arm.upper), 0.0)
    table = 0.0
    if not config.get("allow_object_collisions"):
        table = max(0.0, -minimum_clearance(arm, q, config))
    tip = arm.frames(q)[0][-1]
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    workspace = np.maximum(np.maximum(lo - tip, tip - hi), 0.0)
    return np.r_[joints, table, workspace, self_overlap(arm, q)]


def describe_start_violation(arm, violation):
    joints, table, workspace = violation[:5], violation[5], violation[6:]
    details = [
        f"{name} {np.rad2deg(value):.2f}° beyond its measured limit"
        for name, value in zip(JOINTS, joints)
        if value > 1e-9
    ]
    if table > 1e-9:
        details.append(f"body {table * 1000:.1f} mm inside table clearance")
    if len(violation) > 9 and violation[9] > 1e-9:
        details.append(f"{violation[9] * 1000:.1f} mm model self-overlap")
    for axis, value in zip("XYZ", workspace):
        if value > 1e-9:
            details.append(f"tool {value * 1000:.1f} mm outside workspace {axis}")
    return "; ".join(details)


def recover_start(space, start, targets=(), step_deg=0.5, max_deg=30.0):
    """Shortest straight joint move from an out-of-bounds measured start to a valid pose.

    Every intermediate pose must not worsen any start violation, and must pass
    the self/object collision checks. Nothing is widened: the recovered pose is
    fully valid, and validate_plan re-checks monotonic recovery afterwards.
    """
    arm = space.arm
    start = np.asarray(start, float)
    initial = start_violation(space, start)
    joints, table, workspace = initial[:5], initial[5], initial[6:]
    reason = describe_start_violation(arm, initial)
    if joints.max() > MAX_START_JOINT_EXCESS:
        raise ValueError(
            f"Initial pose: {reason}; more than 2° outside, support the arm and recalibrate"
        )
    if table > MAX_START_TABLE_DEFICIT_M:
        raise ValueError(f"Initial pose: {reason}; support the arm before recovery")
    if workspace[2] > MAX_START_WORKSPACE_Z_M:
        raise ValueError(
            f"Initial pose: {reason}; more than 1.5 cm outside, supported recovery is needed"
        )
    margin = np.deg2rad(0.5)
    inside = np.clip(start, arm.lower + margin, arm.upper - margin)
    neutral = np.clip(np.deg2rad([0, -15, 30, 15, 0]), arm.lower, arm.upper)
    candidates = [inside, *[np.asarray(t, float) for t in targets], neutral]
    locked = space.low == space.high
    best = None
    for goal in candidates:
        goal = np.where(locked, start, goal)
        delta = goal - start
        span = np.max(np.abs(delta))
        if span < 1e-9:
            continue
        unit = delta / span
        previous = initial
        previous_overlap = start_overlap_allowance(arm, start)
        stride = np.deg2rad(step_deg)
        distance = 0.0
        while distance < min(span, np.deg2rad(max_deg)) - 1e-12:
            distance = min(distance + stride, span)
            q = start + unit * distance
            current = start_violation(space, q)
            if (current - previous > 5e-5).any():
                break
            try:
                overlap = self_overlap(arm, q)
                if overlap > previous_overlap + 1e-5:
                    check_self_collision(arm, q)
                previous_overlap = overlap
                if not space.config.get("allow_object_collisions"):
                    from .execution import check_pose_obstacles

                    check_pose_obstacles(arm, q, space.obstacles)
            except ValueError:
                break
            previous = current
            if space.valid(q):
                if best is None or distance < best[0]:
                    best = (distance, q)
                break
    if best is None:
        raise ValueError(
            f"Initial pose: {reason}; no non-worsening recovery within {max_deg:.0f}°"
        )
    return best[1], reason


def timed_joint_plan(arm, path, target, config, info):
    path = np.asarray(path)
    parameter = np.r_[0, np.cumsum(np.max(np.abs(np.diff(path, axis=0)), axis=1))]
    keep = np.r_[True, np.diff(parameter) > 1e-10]
    path, parameter = path[keep], parameter[keep]
    if len(path) < 2:
        path = np.repeat(path, 2, axis=0)
        parameter = np.array([0.0, 1e-10])
    duration = max(
        0.5, 1.875 * parameter[-1] / np.deg2rad(config["max_joint_speed_deg_s"])
    )
    times = np.linspace(0, duration, max(2, int(duration * 20) + 2))
    s = times / duration
    progress = (10 * s**3 - 15 * s**4 + 6 * s**5) * parameter[-1]
    q = np.stack([np.interp(progress, parameter, path[:, i]) for i in range(5)], axis=1)
    plan = Plan(times, q, np.array(target), 0.0, planner_info=[info])
    validate_plan(arm, plan, config)
    from .execution import check_swept_obstacles

    if not config.get("allow_object_collisions"):
        check_swept_obstacles(arm, q, info["obstacles_m"])
    check_self_collision_path(arm, q)
    return plan


def goal_postures(space, arm, start, target, config, goal_pose, candidates):
    """Distinct valid IK postures at the target, nearest to the start first.

    A 5-joint arm reaches a point with a family of wrist pitches and two elbow
    branches; seeding IK across both avoids calling a reachable target blocked
    because the first few nearby postures touch the table or an object.
    """
    from collections import Counter

    reasons = Counter()
    if goal_pose is not None:
        goal = np.asarray(goal_pose, float).copy()
        why = space.reason(goal)
        if why is None:
            return [goal] * max(1, candidates), reasons
        reasons[why] += 1
        return [], reasons
    wanted = max(4, candidates * 2)
    offsets = [
        (0, 0, 0),
        (0, 0, 30),
        (0, 0, -30),
        (0, 0, 60),
        (0, 0, -60),
        (25, -35, 0),
        (-25, 35, 0),
        (25, -35, 45),
        (-25, 35, -45),
        (45, -70, 30),
        (-45, 70, -30),
        (0, 0, 90),
        (0, 0, -90),
        (60, -100, 40),
    ]
    found = []
    for lift, elbow, wrist in offsets:
        guess = np.array(start, float).copy()
        guess[1:4] += np.deg2rad([lift, elbow, wrist])
        guess = np.clip(guess, arm.lower + 1e-7, arm.upper - 1e-7)
        try:
            goal = arm.ik(
                target,
                start,
                config["ik_tolerance_m"],
                config.get("locked_joints", []),
                initial_guess=guess,
            )
        except ValueError as exc:
            reasons[str(exc)] += 1
            continue
        if np.linalg.norm(arm.fk(goal) - target) > config["ik_tolerance_m"]:
            reasons["IK does not reach the selected tip point"] += 1
            continue
        if any(np.max(np.abs(goal - other)) < np.deg2rad(3) for other in found):
            continue
        why = space.reason(goal)
        if why is not None:
            reasons[why] += 1
            continue
        found.append(goal)
        if len(found) >= wanted:
            break
    found.sort(key=lambda g: float(np.max(np.abs(g - np.asarray(start)))))
    return found, reasons


_ROUTE_CACHE = {}
_ROUTE_LOCK = threading.Lock()


def _route_key(arm, start, target, config, obstacles, candidates, revision, goal_pose):
    return (
        id(arm),
        np.asarray(start, float).round(9).tobytes(),
        np.asarray(target, float).round(9).tobytes(),
        json.dumps(config, sort_keys=True, default=str),
        json.dumps(obstacles, sort_keys=True, default=str),
        candidates,
        revision,
        None if goal_pose is None else np.asarray(goal_pose, float).round(9).tobytes(),
    )


def plan_joint_route(
    arm,
    start,
    target,
    config,
    boxes,
    candidates,
    revision,
    requested="rrt",
    guide=None,
    goal_pose=None,
):
    """Whole-arm RRT; identical concurrent requests share one search.

    The path overlay and the selected-route worker both ask for the same RRT
    when RRT is selected; they now wait on one computation instead of
    competing for the interpreter with two identical searches.
    """
    if guide is not None:
        return _plan_joint_route(
            arm,
            start,
            target,
            config,
            boxes,
            candidates,
            revision,
            requested,
            guide,
            goal_pose,
        )
    obstacles = [b.as_dict() if hasattr(b, "as_dict") else b for b in boxes]
    key = _route_key(
        arm, start, target, config, obstacles, candidates, revision, goal_pose
    )
    with _ROUTE_LOCK:
        slot = _ROUTE_CACHE.get(key)
        owner = slot is None
        if owner:
            if len(_ROUTE_CACHE) > 8:
                _ROUTE_CACHE.pop(next(iter(_ROUTE_CACHE)))
            slot = _ROUTE_CACHE[key] = dict(event=threading.Event())
    if not owner:
        while not slot["event"].wait(0.05):
            raise_if_cancelled()
        if "error" in slot:
            if isinstance(slot["error"], PlanningCancelled):
                raise_if_cancelled()
                return plan_joint_route(
                    arm,
                    start,
                    target,
                    config,
                    boxes,
                    candidates,
                    revision,
                    requested,
                    guide,
                    goal_pose,
                )
            raise ValueError(str(slot["error"]))
        plan = copy.deepcopy(slot["plan"])
        for item in plan.planner_info or []:
            if isinstance(item, dict) and "trajectory_method" in item:
                item.update(
                    trajectory_method=requested,
                    fallback=requested != "rrt",
                    joint_space_repair="RRT" if requested != "rrt" else None,
                )
        return plan
    try:
        plan = _plan_joint_route(
            arm,
            start,
            target,
            config,
            boxes,
            candidates,
            revision,
            requested,
            guide,
            goal_pose,
        )
    except ValueError as exc:
        slot["error"] = exc
        if isinstance(exc, PlanningCancelled):
            with _ROUTE_LOCK:
                if _ROUTE_CACHE.get(key) is slot:
                    del _ROUTE_CACHE[key]
        slot["event"].set()
        raise
    slot["plan"] = copy.deepcopy(plan)
    slot["event"].set()
    return plan


def _plan_joint_route(
    arm,
    start,
    target,
    config,
    boxes,
    candidates,
    revision,
    requested="rrt",
    guide=None,
    goal_pose=None,
):
    from .execution import check_pose_obstacles

    obstacles = [b.as_dict() for b in boxes]
    measured = np.asarray(start, float)
    if goal_pose is None:
        from ..reachability import check_reach

        check_reach(arm, target, measured[4], config)
    space = JointSpace(arm, config, obstacles, measured)
    recovery = None
    if not space.valid(measured):
        if not config.get("allow_object_collisions"):
            check_pose_obstacles(arm, measured, obstacles)
        start_overlap_allowance(arm, measured)
        # Joint-stop noise, a sub-millimetre hull overlap at rest or a slightly low parked tool: plan a short,
        # monotonic recovery move first instead of refusing every route.
        recovery, recovery_reason = recover_start(
            space, measured, [goal_pose] if goal_pose is not None else []
        )
        start = recovery
    best = None
    reasons = []
    goals, goal_reasons = goal_postures(
        space, arm, start, target, config, goal_pose, candidates
    )
    for index, goal in enumerate(goals):
        try:
            rng = np.random.default_rng(1729 + index)
            if guide is not None:
                base_rng = rng

                class Guided:
                    def uniform(self, low, high):
                        if base_rng.random() < 0.5:
                            return np.clip(
                                guide[base_rng.integers(len(guide))]
                                + base_rng.normal(0, 0.08, 5),
                                low,
                                high,
                            )
                        return base_rng.uniform(low, high)

                rng = Guided()
            path = rrt_connect(space, start, goal, rng, step=0.15, max_iterations=500)
            if path is None and index == len(goals) - 1 and best is None:
                # Last chance before reporting failure: a deeper search on the
                # same goal, still inside the shared JointSpace time budget.
                path = rrt_connect(
                    space,
                    start,
                    goals[0],
                    np.random.default_rng(2729),
                    step=0.15,
                    max_iterations=3000,
                )
            if path is None:
                continue
            path = shortcut(
                space, path, np.random.default_rng(1729 + index), attempts=15
            )
            if recovery is not None:
                path = np.vstack((measured, path))
            info = dict(
                trajectory_method=requested,
                candidates=candidates,
                scene_revision=revision,
                obstacles_m=obstacles,
                planning_space="joint",
                collision_predictions=[],
                allow_object_collisions=bool(config.get("allow_object_collisions")),
                fallback=requested != "rrt",
                joint_space_repair="RRT" if requested != "rrt" else None,
                collision_state_checks=space.checks,
                start_recovery=None
                if recovery is None
                else dict(
                    reason=recovery_reason,
                    recovered_pose_deg=np.rad2deg(recovery).tolist(),
                ),
            )
            plan = timed_joint_plan(arm, path, target, config, info)
            if best is None or plan.time_s[-1] < best.time_s[-1]:
                best = plan
            if index + 1 >= candidates:
                break
        except PlanningCancelled:
            raise
        except ValueError as exc:
            reasons.append(str(exc))
    if best is not None:
        return best
    if not goals:
        summary = "; ".join(
            f"{reason} ({count}/{sum(goal_reasons.values())} IK postures)"
            for reason, count in goal_reasons.most_common(2)
        )
        raise ValueError(
            "No whole-arm joint-space route: every target posture is blocked"
            + (f": {summary}" if summary else "")
        )
    raise ValueError(
        "No whole-arm joint-space route"
        + (
            ": " + reasons[-1]
            if reasons
            else f": RRT found no connection to {len(goals)} valid target postures"
        )
    )
