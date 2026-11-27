"""Lift a selected Cartesian route to checked, timed numerical joint IK."""

import time

import numpy as np

from ..planning import Plan, validate_plan
from ..test_bed import table_height_at


def route_info(plan):
    return next((i for i in plan.planner_info or [] if "trajectory_method" in i), None)


def same_geometry(expected, actual, tolerance=0.02):
    from scipy.optimize import linear_sum_assignment

    if len(expected) != len(actual):
        return False
    if not expected:
        return True
    a = np.array([np.r_[b["low_m"], b["high_m"]] for b in expected])
    b = np.array([np.r_[b["low_m"], b["high_m"]] for b in actual])
    costs = np.max(abs(a[:, None] - b[None]), axis=-1)
    rows, cols = linear_sum_assignment(costs)
    return bool((costs[rows, cols] <= tolerance).all())


def require_same_scene(route, live, arm=None, joints=None):
    report = live.get("obstacle_detection") or {}
    if (
        time.time() - live.get("unix_time_s", 0) > 2
        or live.get("camera_stale")
        or report.get("error")
    ):
        raise RuntimeError("Scene monitoring unavailable; position hold retained")
    if route.get("allow_object_collisions"):
        return
    expected = route["obstacles_m"]
    actual = live.get("obstacle_boxes", [])
    if not same_geometry(expected, actual):
        if arm is None or joints is None:
            raise RuntimeError("Scene geometry changed; refresh the path preview")
        check_swept_obstacles(arm, joints, actual)
    # Revisions and detector ordering are not evidence that a route is blocked.
    from copy import deepcopy

    route["obstacles_m"] = deepcopy(actual)
    route["scene_revision"] = report.get("scene_revision", route.get("scene_revision"))


def segment_box(a, b, low, high):
    enter, leave = 0.0, 1.0
    for axis in range(3):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if not low[axis] <= a[axis] <= high[axis]:
                return False
        else:
            near, far = sorted(
                ((low[axis] - a[axis]) / delta, (high[axis] - a[axis]) / delta)
            )
            enter, leave = max(enter, near), min(leave, far)
            if enter > leave:
                return False
    return True


def segment_box_distance(a, b, low, high):
    """Exact Euclidean segment/AABB distance via piecewise quadratic minima."""
    a, b, low, high = map(np.asarray, (a, b, low, high))
    delta = b - a
    times = [0.0, 1.0]
    for j in range(3):
        if abs(delta[j]) > 1e-12:
            times.extend(
                t
                for t in ((low[j] - a[j]) / delta[j], (high[j] - a[j]) / delta[j])
                if 0 < t < 1
            )
    times = np.unique(times)
    best = float("inf")
    for left, right in zip(times[:-1], times[1:]):
        middle = a + ((left + right) / 2) * delta
        active = (middle < low) | (middle > high)
        boundary = np.where(middle < low, low, high)
        linear = delta[active]
        offset = (a - boundary)[active]
        optimum = np.clip(-offset @ linear / max(linear @ linear, 1e-30), left, right)
        point = a + optimum * delta
        best = min(best, float(np.linalg.norm(point - np.clip(point, low, high))))
    return best


class ArmCollision(ValueError):
    def __init__(self, link, obstacle, distance, radius, tip):
        self.link, self.obstacle = link, obstacle
        self.distance_m, self.radius_m = distance, radius
        self.tip_m = np.asarray(tip).tolist()
        self.path_fraction = None
        super().__init__(
            f"Arm link {link + 1} overlaps object {obstacle + 1} clearance: {distance * 1000:.1f} mm, "
            f"needs {radius * 1000:.0f} mm"
        )


def check_pose_obstacles(arm, pose, obstacles):
    if not obstacles:
        return
    points = arm.frames(pose)[0]
    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        # Capsule/AABB distance, with an inexpensive broad-phase screen. Includes
        # base links and fixed tool offset; not a visual mesh collision model.
        padding = 0.045 if index == len(points) - 2 else 0.035
        for box_index, box in enumerate(obstacles):
            if segment_box(
                a,
                b,
                np.array(box["low_m"]) - padding,
                np.array(box["high_m"]) + padding,
            ):
                distance = segment_box_distance(
                    a, b, np.array(box["low_m"]), np.array(box["high_m"])
                )
                if distance <= padding:
                    raise ArmCollision(index, box_index, distance, padding, points[-1])


def check_swept_obstacles(arm, joints, obstacles):
    for index, (a, b) in enumerate(zip(joints[:-1], joints[1:])):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(0.5))))
        for t in np.linspace(0, 1, count + 1):
            try:
                check_pose_obstacles(arm, a * (1 - t) + b * t, obstacles)
            except ArmCollision as exc:
                exc.path_fraction = (index + t) / max(len(joints) - 1, 1)
                exc.args = (str(exc) + f" at {exc.path_fraction * 100:.0f}% of route",)
                raise


def execution_scene(config, boxes, arm):
    from .live_preview import preview_scene

    scene = preview_scene(config, boxes)
    # The tool-offset segment itself is padded by 45 mm in the arm checker.
    # Account for that segment's offset from the tip when routing the tip.
    clearance = 0.045 + np.linalg.norm(getattr(arm, "tool_offset", np.zeros(3))) + 0.002
    scene.tool_radius = max(scene.tool_radius, clearance - scene.margin)
    return scene


# Intermediate learned-route waypoints only guide the tool; every resulting
# joint pose is still checked for table, workspace, object and self clearance.
WAYPOINT_IK_TOLERANCE_M = 0.008


def waypoint_ik(arm, point, seed, config, final=False):
    """Sequential route IK: exact at the target, near-exact between waypoints.

    A route point a few millimetres outside the locked-roll reach (or a
    local minimum from the previous pose) used to reject the whole learned
    candidate and force an RRT fallback.
    """
    strict = config["ik_tolerance_m"]
    locked = config.get("locked_joints", [])
    try:
        return arm.ik(point, seed, strict, locked_joints=locked)
    except ValueError as first:
        failure = first
    guesses = []
    for offset in (20, -20, 45, -45):
        guess = np.array(seed, float).copy()
        guess[3] += np.deg2rad(offset)
        guesses.append(np.clip(guess, arm.lower + 1e-7, arm.upper - 1e-7))
    for tolerance in (
        (strict,) if final else (strict, max(strict, WAYPOINT_IK_TOLERANCE_M))
    ):
        for guess in guesses if tolerance == strict else [None, *guesses]:
            try:
                return arm.ik(
                    point, seed, tolerance, locked_joints=locked, initial_guess=guess
                )
            except ValueError:
                continue
    raise failure


def lift_route(
    arm, start, target, xy, config, *, method, candidates, boxes, scene_revision
):
    start, target, xy = np.array(start), np.array(target), np.asarray(xy, float)
    tip = arm.fk(start)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("Planner returned a malformed route")
    if not np.allclose(xy[0], tip[:2], atol=1e-6) or not np.allclose(
        xy[-1], target[:2], atol=1e-6
    ):
        raise ValueError("Route endpoints do not match measured start and target")
    # Densify Cartesian segments before sequential, nearby-solution numerical IK.
    dense = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.005)))
        dense.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    dense = np.array(dense)
    if np.linalg.norm(target[:2] - tip[:2]) < 1e-8:
        dense = np.repeat(
            tip[None, :2], max(2, int(abs(target[2] - tip[2]) / 0.005) + 2), axis=0
        )
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    fractions = lengths / max(lengths[-1], 1e-12)
    if lengths[-1] < 1e-8:
        fractions = np.linspace(0, 1, len(dense))
    floor = np.asarray(table_height_at(dense, config))
    initial_hover = tip[2] - float(table_height_at(tip[:2], config))
    final_hover = target[2] - float(table_height_at(target[:2], config))
    xyz = np.c_[
        dense, floor + initial_hover * (1 - fractions) + final_hover * fractions
    ]
    xyz[0], xyz[-1] = tip, target
    orientation = config.get("grasp_axis_direction")
    if orientation is not None:
        refined = [xyz[0]]
        for a, b in zip(xyz[:-1], xyz[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.005)))
            refined.extend(a + (b - a) * t for t in np.linspace(0, 1, count + 1)[1:])
        xyz = np.array(refined)
    from .cancel import raise_if_cancelled

    poses = [start]
    for point in xyz[1:]:
        raise_if_cancelled()
        if orientation is not None:
            from ..grasp_geometry import oriented_ik

            poses.append(
                oriented_ik(
                    arm,
                    point,
                    poses[-1],
                    orientation,
                    config["ik_tolerance_m"],
                    config.get("locked_joints", []),
                )
            )
        else:
            poses.append(
                waypoint_ik(
                    arm,
                    point,
                    poses[-1],
                    config,
                    final=len(poses) == len(xyz) - 1,
                )
            )
    poses = np.array(poses)
    waypoint_error = max(
        float(np.linalg.norm(arm.fk(q) - p)) for q, p in zip(poses[1:], xyz[1:])
    )
    parameter = np.r_[0, np.cumsum(np.max(np.abs(np.diff(poses, axis=0)), axis=1))]
    keep = np.r_[True, np.diff(parameter) > 1e-10]
    poses, parameter = poses[keep], parameter[keep]
    if len(poses) < 2:
        poses = np.repeat(poses, 2, axis=0)
        parameter = np.array([0.0, 1e-10])
    duration = max(
        0.5, 1.875 * parameter[-1] / np.deg2rad(config["max_joint_speed_deg_s"])
    )
    times = np.linspace(0, duration, max(2, int(np.ceil(duration * 20)) + 1))
    s = times / duration
    progress = (10 * s**3 - 15 * s**4 + 6 * s**5) * parameter[-1]
    joints = np.stack(
        [np.interp(progress, parameter, poses[:, j]) for j in range(5)], axis=1
    )
    info = dict(
        trajectory_method=method,
        candidates=candidates,
        scene_revision=scene_revision,
        obstacles_m=[b.as_dict() for b in boxes],
        cartesian_route_m=xyz.tolist(),
        fallback=False,
        allow_object_collisions=bool(config.get("allow_object_collisions", False)),
        collision_predictions=[],
        max_waypoint_ik_error_m=waypoint_error,
    )
    plan = Plan(times, joints, target, 0.0, planner_info=[info])
    if orientation is not None:
        from ..grasp_geometry import tilt_degrees, tool_geometry

        info["grasp_axis_direction"] = list(orientation)
        info["grasp_tilt_deg"] = tilt_degrees(arm, joints[-1], -np.asarray(orientation))
        info["max_grasp_tilt_deg"] = max(
            tilt_degrees(arm, q, -np.asarray(orientation)) for q in joints
        )
        info["grasp_wrist_m"] = tool_geometry(arm, joints[-1])[1].tolist()
    validate_plan(arm, plan, config)
    # Validate dense joint interpolation too: waypoint-only checks miss collisions.
    dense_joints = [joints[0]]
    for a, b in zip(joints[:-1], joints[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(0.5))))
        dense_joints.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    dense_joints = np.array(dense_joints)
    from .joint_planner import check_self_collision_path

    check_self_collision_path(arm, dense_joints)
    dense_times = np.r_[
        0,
        np.cumsum(
            np.maximum(
                np.max(np.abs(np.diff(dense_joints, axis=0)), axis=1)
                / np.deg2rad(config["max_joint_speed_deg_s"]),
                1e-6,
            )
        ),
    ]
    validate_plan(arm, Plan(dense_times, dense_joints, target, 0.0), config)
    try:
        check_swept_obstacles(arm, joints, info["obstacles_m"])
    except ArmCollision as exc:
        if not info["allow_object_collisions"]:
            exc.plan = plan
            raise
        info["collision_predictions"].append(
            dict(
                link=exc.link,
                object=exc.obstacle,
                distance_m=exc.distance_m,
                radius_m=exc.radius_m,
                tip_m=exc.tip_m,
                path_fraction=exc.path_fraction,
                message=str(exc),
            )
        )
    # The numerical joint interpolation must also keep the tool in the XY corridor.
    scene = execution_scene(config, boxes, arm)
    if not scene.path_free(np.array([arm.fk(p)[:2] for p in dense_joints])):
        if not info["allow_object_collisions"]:
            raise ValueError("Joint interpolation leaves the collision-free tool route")
        info["tool_route_collision"] = True
    return plan


class RoutePlanner:
    def __init__(self):
        self.suite = None

    def plan(self, arm, start, target, config, boxes, method, candidates, revision):
        from .evaluate import classical
        from .live_preview import preview_scene, shared_suite

        scene = execution_scene(config, boxes, arm)
        initial = arm.fk(start)[:2]
        override = bool(config.get("allow_object_collisions", False))
        if method == "rrt" and not override:
            from .joint_planner import plan_joint_route

            return plan_joint_route(
                arm, start, target, config, boxes, candidates, revision
            )
        if method == "direct":
            paths = [np.linspace(initial, target[:2], 32)]
        elif method == "rrt":
            paths = [
                classical(
                    scene,
                    dict(
                        index=1729 + i, start=initial.tolist(), goal=target[:2].tolist()
                    ),
                )
                for i in range(candidates)
            ]
            paths = [p for p in paths if p is not None]
            if not paths and override:
                from dataclasses import replace

                unblocked = replace(scene, boxes=(), circles=())
                paths = [
                    classical(
                        unblocked,
                        dict(
                            index=1729, start=initial.tolist(), goal=target[:2].tolist()
                        ),
                    )
                ]
                paths = [p for p in paths if p is not None]
        else:
            if self.suite is None:
                self.suite = shared_suite()
            row = next(
                r
                for r in self.suite.compare(
                    preview_scene(config, boxes),
                    initial,
                    target[:2],
                    candidates,
                    allow_collision=override,
                    methods=(method,),
                )
                if r["method"] == method
            )
            paths = [np.array(p) for p in row["raw_paths"]]
        paths.sort(key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        reasons = []
        for path in paths:
            try:
                from .route_refinement import refine_route

                raw_path = path.copy()
                if override:
                    refinement = dict(applied=False, max_shift_m=0.0)
                else:
                    path, refinement = refine_route(path, scene)
                plan = lift_route(
                    arm,
                    start,
                    target,
                    path,
                    config,
                    method=method,
                    candidates=1 if method == "mlp" else candidates,
                    boxes=boxes,
                    scene_revision=revision,
                )
                route_info(plan).update(
                    path_refinement=refinement,
                    raw_path_xy_m=raw_path.tolist(),
                    tool_clearance_m=scene.inflation,
                )
                return plan
            except ValueError as exc:
                if (
                    isinstance(exc, ArmCollision)
                    and hasattr(exc, "plan")
                    and not override
                ):
                    from .joint_planner import plan_joint_route

                    try:
                        repaired = plan_joint_route(
                            arm,
                            start,
                            target,
                            config,
                            boxes,
                            candidates,
                            revision,
                            requested=method,
                            guide=exc.plan.joints_rad,
                        )
                        route_info(repaired)["raw_path_xy_m"] = raw_path.tolist()
                        return repaired
                    except ValueError as repair_error:
                        reasons.append(str(repair_error))
                reasons.append(str(exc))
        if method in ("mlp", "fm", "mf", "imf") and not override:
            from .joint_planner import plan_joint_route

            repaired = plan_joint_route(
                arm,
                start,
                target,
                config,
                boxes,
                candidates,
                revision,
                requested=method,
            )
            route_info(repaired)["fallback_reason"] = (
                "All learned candidates rejected: " + ("; ".join(reasons[:2]))
            )
            return repaired
        raise ValueError(
            f"{method.upper()}: no executable candidate"
            + (": " + reasons[0] if reasons else " (all paths rejected)")
        )
