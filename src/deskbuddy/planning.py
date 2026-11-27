"""The baseline and future learned planner share this validation boundary."""

from dataclasses import dataclass

import numpy as np

from .test_bed import table_height_at


@dataclass
class Plan:
    time_s: np.ndarray
    joints_rad: np.ndarray
    target_m: np.ndarray
    endpoint_error_m: float
    recovery_waypoint_m: np.ndarray | None = None
    planner_info: list | None = None


def hover_bounds(surface, config):
    """Intersect hover heights with the same target workspace used by clicks."""
    surface = np.asarray(surface, float)
    if surface.shape != (3,) or not np.isfinite(surface).all():
        raise ValueError("Selected surface must be a finite XYZ point")
    floor = max(
        config["workspace_min_m"][2],
        float(table_height_at(surface[:2], config))
        + config["link_clearance_m"]
        + 0.002,
    )
    minimum = max(0.02, floor - float(surface[2]))
    maximum = min(0.20, config["workspace_max_m"][2] - float(surface[2]))
    if minimum > maximum:
        raise ValueError("Selected surface cannot be approached within the hover range")
    check_workspace(surface + [0, 0, minimum], config)
    check_workspace(surface + [0, 0, maximum], config)
    return minimum, maximum


def halve_steps(plan):
    """Two equal angular substeps without stretching trajectory duration.

    The 250 Hz controller interpolates this reference independently. The physical
    reference remains continuous; tracking error is never reset per substep.
    """
    q = np.asarray(plan.joints_rad)
    times = np.asarray(plan.time_s)
    refined = np.empty((2 * len(q) - 1, 5))
    refined[::2] = q
    refined[1::2] = (q[:-1] + q[1:]) / 2
    refined_time = np.empty(2 * len(times) - 1)
    refined_time[::2] = times
    refined_time[1::2] = (times[:-1] + times[1:]) / 2
    return Plan(
        refined_time,
        refined,
        np.asarray(plan.target_m).copy(),
        plan.endpoint_error_m,
        plan.recovery_waypoint_m,
        plan.planner_info,
    )


def check_workspace(point, config, label="Selected target"):
    p = np.asarray(point, float)
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError(f"{label} must be a finite XYZ point")
    failures = [
        f"{axis}={value * 100:.1f} cm (allowed {low * 100:.1f}–{high * 100:.1f} cm)"
        for axis, value, low, high in zip("xyz", p, lo, hi)
        if value < low or value > high
    ]
    if failures:
        raise ValueError(f"{label} outside workspace: " + "; ".join(failures))
    if label == "Selected target" and config.get("restrict_to_test_bed", True):
        from .test_bed import contains_xy

        if not contains_xy(p, config):
            raise ValueError("Selected target outside the inset A4 test bed")


def validate_plan(arm, plan, config):
    q, times = np.asarray(plan.joints_rad), np.asarray(plan.time_s)
    if q.ndim != 2 or q.shape[1] != 5 or len(q) < 2 or times.shape != (len(q),):
        raise ValueError("Malformed plan")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Plan contains nonfinite values or invalid timestamps")
    excess = np.maximum(np.maximum(arm.lower - q, q - arm.upper), 0.0)
    tolerance = 1e-8  # radians: numerical precision, not additional travel
    if excess.max() > tolerance:
        growth = np.diff(excess, axis=0)
        if excess[0].max() > np.deg2rad(2.0) + tolerance or (growth > tolerance).any():
            from .kinematics import JOINTS

            if excess[0].max() > np.deg2rad(2.0) + tolerance:
                row, joint = 0, int(np.argmax(excess[0]))
                reason = "measured start exceeds recovery allowance of 2 degrees"
            else:
                step, joint = np.unravel_index(np.argmax(growth), growth.shape)
                row = step + 1
                reason = "path moves farther outside limit"
            raise ValueError(
                f"Joint limit violated: {JOINTS[joint]} at step {row}, "
                f"{np.rad2deg(q[row, joint]):.3f} deg "
                f"(allowed {np.rad2deg(arm.lower[joint]):.3f} to {np.rad2deg(arm.upper[joint]):.3f}); "
                f"excess {np.rad2deg(excess[row, joint]):.4f} deg; {reason}"
            )
    velocities = np.abs(np.diff(q, axis=0) / np.diff(times)[:, None])
    if velocities.max() > np.deg2rad(config["max_joint_speed_deg_s"]) * 1.001:
        raise ValueError("Joint velocity limit violated")
    from .table_collision import clearances, rejection

    tool_positions = []
    link_clearances = []
    for pose in q:
        points, _ = arm.frames(pose)
        link_clearances.append([value for _, value in clearances(arm, pose, config)])
        tool_positions.append(points[-1])
    deficit = np.maximum(-np.array(link_clearances), 0.0)
    override = config.get("allow_object_collisions", False)
    if deficit.max() > 1e-8 and override:
        index = int(np.argmax(deficit.max(axis=1)))
        entry = dict(
            table_collision_prediction=True,
            pose_rad=q[index].tolist(),
            message=rejection(arm, q[index], config, "Planned path"),
        )
        plan.planner_info = (plan.planner_info or []) + [entry]
    if not override and (deficit[0].max() > 0.005 or deficit[-1].max() > 1e-8):
        index = 0 if deficit[0].max() > 0.005 else len(q) - 1
        raise ValueError(rejection(arm, q[index], config, "Planned path"))
    if not override and deficit[0].max() > 1e-8:
        if (np.diff(deficit, axis=0) > 0.00005).any():
            raise ValueError("Recovery path reduces table clearance")
    elif not override and deficit.max() > 1e-8:
        index = int(np.argmax(deficit.max(axis=1)))
        raise ValueError(rejection(arm, q[index], config, "Planned path"))
    check_workspace(plan.target_m, config)
    tools = np.array(tool_positions)
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    violation = np.maximum(np.maximum(lo - tools, tools - hi), 0)
    if violation[0].max() > 1e-8:
        if violation[0, 2] > 0.015:
            raise ValueError(
                "Starting tool height is more than 1.5 cm outside the workspace; supported recovery is needed"
            )
        if (np.diff(violation, axis=0) > 0.00005).any():
            raise ValueError(
                "Recovery path moves farther outside the workspace; choose a simpler recovery"
            )
    elif violation.max() > 1e-8:
        index = int(np.argmax(violation.max(axis=1)))
        check_workspace(tools[index], config, label="Planned path")
    error = float(np.linalg.norm(arm.fk(q[-1]) - plan.target_m))
    if not np.isfinite(error) or error > config["ik_tolerance_m"]:
        raise ValueError("Endpoint outside IK tolerance")
    plan.endpoint_error_m = error
    return plan


def plan_reach(arm, start, target, config):
    check_workspace(target, config)
    start = np.asarray(start, float)
    from .reachability import check_reach

    check_reach(arm, target, start[4], config)
    if (start < arm.lower).any() or (start > arm.upper).any():
        raise ValueError(arm.limit_error(start))
    if config.get("validated_wrist_roll_deg") is not None:
        difference = (
            np.rad2deg(start[4]) - config["validated_wrist_roll_deg"] + 180
        ) % 360 - 180
        if abs(difference) > config["wrist_roll_tolerance_deg"]:
            raise ValueError(
                "Return wrist roll to its calibrated forward orientation before previewing this reach"
            )
    points, _ = arm.frames(start)
    from .table_collision import minimum_clearance

    clearance_deficit = max(
        0.0,
        -minimum_clearance(arm, start, config),
    )
    if config.get("allow_object_collisions"):
        clearance_deficit = 0.0
    if clearance_deficit > 0.005:
        raise ValueError(
            "Current arm pose is below table clearance; support the arm before recovery"
        )
    initial = points[-1]
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    outside = np.maximum(np.maximum(lo - initial, initial - hi), 0)
    recovery = None
    targets = []
    if outside.max() > 1e-8 or clearance_deficit > 1e-8:
        if outside[2] > 0.015:
            raise ValueError(
                "Starting tool height is more than 1.5 cm outside the workspace; supported recovery is needed"
            )
        recovery = initial.copy()
        recovery[initial < lo] = lo[initial < lo] + 0.005
        recovery[initial > hi] = hi[initial > hi] - 0.005
        if clearance_deficit > 1e-8:
            recovery[2] = max(recovery[2], initial[2] + clearance_deficit + 0.003)
        targets.append(recovery)
    targets.append(np.asarray(target))
    times = []
    poses = []
    elapsed = 0.0
    current = start
    diagnostics = []
    for waypoint in targets:
        segment_time, segment_q = _joint_segment(
            arm, current, waypoint, config, diagnostics
        )
        skip = 1 if poses else 0
        times.extend((elapsed + segment_time[skip:]).tolist())
        poses.extend(segment_q[skip:])
        elapsed += segment_time[-1]
        current = segment_q[-1]
    try:
        return validate_plan(
            arm,
            Plan(
                np.array(times),
                np.array(poses),
                np.asarray(target),
                0.0,
                recovery,
                diagnostics,
            ),
            config,
        )
    except ValueError as exc:
        if config.get("planner_mode") != "flow_assisted":
            raise
        fallback = plan_reach(
            arm, start, target, dict(config, planner_mode="numerical")
        )
        fallback.planner_info = [
            dict(
                requested="flow_assisted",
                used="numerical",
                fallback_reason=f"Learned-seeded path rejected: {exc}",
            )
        ]
        return fallback


def _joint_segment(arm, start, target, config, diagnostics=None):
    from .learned.solver import solve_goal

    goal, info = solve_goal(arm, start, target, config)
    if diagnostics is not None:
        diagnostics.append(info)
    return interpolate_joints(start, goal, config)


def interpolate_joints(start, goal, config):
    """Time a known joint-space segment; callers must validate the complete path."""
    # Joint-space quintic interpolation: zero velocity and acceleration at each end.
    # Duration accounts for the peak derivative of 10s^3 - 15s^4 + 6s^5 (1.875).
    duration = max(
        0.6,
        1.9
        * float(np.max(np.abs(goal - start)))
        / np.deg2rad(config["max_joint_speed_deg_s"]),
    )
    times = np.linspace(0, duration, int(np.ceil(duration * config["control_hz"])) + 1)
    s = times / duration
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    q = start + blend[:, None] * (goal - start)
    return times, q
