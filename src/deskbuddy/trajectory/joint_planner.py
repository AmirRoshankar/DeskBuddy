"""Configuration-space RRT-Connect with whole-arm state/edge validation."""

import time

import numpy as np

from ..kinematics import JOINTS
from ..planning import Plan, validate_plan
from ..table_collision import minimum_clearance
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
        from .execution import check_pose_obstacles

        key = tuple(np.round(q, 7))
        if key in self.cache:
            return self.cache[key]
        self.checks += 1
        valid = True
        try:
            if (q < self.low - 1e-8).any() or (q > self.high + 1e-8).any():
                raise ValueError("Joint bounds")
            points = self.arm.frames(q)[0]
            if (
                not self.config.get("allow_object_collisions")
                and minimum_clearance(self.arm, q, self.config) < -1e-8
            ):
                raise ValueError("Table clearance")
            tip = points[-1]
            if (tip < self.config["workspace_min_m"]).any() or (
                tip > self.config["workspace_max_m"]
            ).any():
                raise ValueError("Workspace")
            check_self_collision(self.arm, q)
            if not self.config.get("allow_object_collisions"):
                check_pose_obstacles(self.arm, q, self.obstacles)
        except ValueError:
            valid = False
        self.cache[key] = valid
        return valid

    def segment_free(self, a, b):
        if time.monotonic() > self.deadline:
            raise ValueError("Whole-arm RRT time budget reached")
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(1))))
        return all(self.valid(a + (b - a) * t) for t in np.linspace(0, 1, count + 1))

    def path_free(self, path):
        return all(self.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))


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
    for pose in q:
        check_self_collision(arm, pose)
    return plan


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
    from .execution import check_pose_obstacles

    obstacles = [b.as_dict() for b in boxes]
    space = JointSpace(arm, config, obstacles, start)
    if not space.valid(np.asarray(start)):
        if not config.get("allow_object_collisions"):
            check_pose_obstacles(arm, start, obstacles)
        check_self_collision(arm, start)
        raise ValueError("Initial pose violates table or workspace constraints")
    best = None
    reasons = []
    for index in range(
        max(1, candidates) if goal_pose is not None else max(4, candidates * 2)
    ):
        guess = np.array(start).copy()
        guess[3] = np.clip(
            guess[3] + np.deg2rad([0, 30, -30, 60, -60, 15, -15, 0][index % 8]),
            arm.lower[3] + 1e-7,
            arm.upper[3] - 1e-7,
        )
        try:
            goal = (
                np.asarray(goal_pose).copy()
                if goal_pose is not None
                else arm.ik(
                    target,
                    start,
                    config["ik_tolerance_m"],
                    config.get("locked_joints", []),
                    initial_guess=guess,
                )
            )
            if np.linalg.norm(arm.fk(goal) - target) > config["ik_tolerance_m"]:
                raise ValueError("Goal pose does not reach selected tip point")
            if not space.valid(goal):
                continue
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
            if path is None:
                continue
            path = shortcut(
                space, path, np.random.default_rng(1729 + index), attempts=15
            )
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
            )
            plan = timed_joint_plan(arm, path, target, config, info)
            if best is None or plan.time_s[-1] < best.time_s[-1]:
                best = plan
            if index + 1 >= candidates:
                break
        except ValueError as exc:
            reasons.append(str(exc))
    if best is not None:
        return best
    raise ValueError(
        "No whole-arm joint-space route"
        + (": " + reasons[-1] if reasons else "; target posture is blocked")
    )
