"""Table-relative grasp poses: hard TCP position, best feasible downward tool axis."""

import numpy as np
from scipy.optimize import least_squares, minimize

from .kinematics import JOINTS
from .test_bed import corners, table_height_at


def table_normal(config):
    bed = corners(config, inset=False)
    normal = (
        np.array([0.0, 0.0, 1.0])
        if bed is None
        else np.cross(bed[1] - bed[0], bed[3] - bed[0])
    )
    normal = normal / np.linalg.norm(normal)
    return normal if normal[2] > 0 else -normal


def tool_geometry(arm, q):
    points, _ = arm.frames(q)
    mount = next(i + 1 for i, j in enumerate(arm.chain) if j[0] == "wrist_roll")
    nominal_tip = next(
        i + 1 for i, j in enumerate(arm.chain) if j[0] == "gripper_frame_joint"
    )
    vector = points[nominal_tip] - points[mount]
    return points[-1], points[mount], vector / np.linalg.norm(vector)


def tilt_degrees(arm, q, normal):
    axis = tool_geometry(arm, q)[2]
    return float(np.rad2deg(np.arccos(np.clip(axis @ (-np.asarray(normal)), -1, 1))))


def oriented_ik(
    arm, target, seed, direction, tolerance=0.002, locked_joints=(), thorough=False
):
    """Constrain actual calibrated tip XYZ; minimize physical gripper-axis tilt.

    Full orientation (finger yaw) is deliberately free. A hard position constraint
    prevents trading tabletop penetration for a smaller orientation residual.
    """
    target, seed = np.asarray(target, float), np.asarray(seed, float)
    direction = np.asarray(direction, float)
    if (
        target.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite([target, direction]).all()
        or np.linalg.norm(direction) < 1e-8
    ):
        raise ValueError("Finite grasp position and approach direction required")
    direction = direction / np.linalg.norm(direction)
    arm.frames(seed)
    if (seed < arm.lower - 1e-8).any() or (seed > arm.upper + 1e-8).any():
        raise ValueError(arm.limit_error(seed, "Grasp IK seed"))
    seed = np.clip(seed, arm.lower, arm.upper)
    free = np.array(
        [i for i, j in enumerate(JOINTS) if j not in locked_joints], dtype=int
    )
    if len(free) < 3:
        raise ValueError("Grasp pose needs at least three free arm joints")

    def expand(x):
        q = seed.copy()
        q[free] = x
        return q

    def weighted(x):
        q = expand(x)
        tip, _, axis = tool_geometry(arm, q)
        return np.r_[tip - target, 0.1 * (axis - direction), 0.00001 * (q - seed)]

    def objective(x):
        q = expand(x)
        axis = tool_geometry(arm, q)[2]
        return 1 - float(axis @ direction) + 1e-7 * np.sum((q - seed) ** 2)

    def equality(x):
        return (arm.fk(expand(x)) - target) * 10

    best = None
    guesses = [seed]
    if thorough:
        for flex, roll in [(35, 0), (-35, 0), (0, 90), (0, -90), (60, 90), (-60, -90)]:
            guess = seed.copy()
            guess[3] += np.deg2rad(flex)
            guess[4] += np.deg2rad(roll)
            guesses.append(guess)
    for guess in guesses:
        initial = np.clip(guess[free], arm.lower[free] + 1e-8, arm.upper[free] - 1e-8)
        fit = least_squares(
            weighted,
            initial,
            bounds=(arm.lower[free], arm.upper[free]),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=100,
        )
        q = expand(fit.x)
        gap = np.linalg.norm(arm.fk(q) - target)
        angle = np.arccos(np.clip(tool_geometry(arm, q)[2] @ direction, -1, 1))
        if gap < min(tolerance, 0.00001) and angle < np.deg2rad(0.2):
            return q
        # The orientation-weighted fit can end on an unreachable position branch.
        # Always retain a position-feasible start independently of that fit.
        feasible = None
        for start in (guess, q):
            try:
                feasible = arm.ik(
                    target, seed, tolerance, locked_joints, initial_guess=start
                )
                break
            except ValueError:
                pass
        if feasible is None:
            continue
        result = minimize(
            objective,
            feasible[free],
            method="SLSQP",
            bounds=list(zip(arm.lower[free], arm.upper[free])),
            constraints=[dict(type="eq", fun=equality)],
            options=dict(maxiter=120, ftol=1e-10),
        )
        for candidate in (feasible, expand(result.x), q):
            tip, _, axis = tool_geometry(arm, candidate)
            if (
                np.linalg.norm(tip - target) > min(tolerance, 0.00001)
                or axis @ direction <= 0
            ):
                continue
            score = (1 - axis @ direction, np.linalg.norm(candidate - seed))
            if best is None or score < best[0]:
                best = (score, candidate.copy())
    if best is None:
        raise ValueError(
            "No downward grasp pose reaches this tip point within joint limits"
        )
    return best[1]


def grasp_settings(arm, config, source, hover):
    import copy

    normal = table_normal(config)
    source = np.asarray(source, float).copy()
    bed = config.get("test_bed")
    margin = bed.get("clearance_margin_m", 0.003) if bed else 0.0
    source[2] = float(table_height_at(source[:2], config)) - margin
    contact = source + normal * 0.005
    floor = float(config["workspace_min_m"][2])
    if contact[2] < floor:
        contact += normal * ((floor - contact[2]) / normal[2])
    if contact[2] > config["workspace_max_m"][2]:
        raise ValueError("Grasp contact is above the configured Z maximum")
    height = max(0.02, float((np.asarray(hover) - contact) @ normal))
    above = contact + normal * height
    aligned = copy.deepcopy(config)
    aligned["locked_joints"] = [
        j for j in config.get("locked_joints", []) if j != "wrist_roll"
    ]
    aligned["grasp_axis_direction"] = (-normal).tolist()
    touching = copy.deepcopy(aligned)
    touching["grasp_contact_mode"] = True
    # Grasping respects the operator floor; no hidden downward workspace override.
    return normal, contact, above, aligned, touching


def attach_grasp_preview(plan, arm, config, source):
    normal, contact, above, aligned, _ = grasp_settings(
        arm, config, source, plan.target_m
    )
    q = oriented_ik(
        arm,
        contact,
        plan.joints_rad[-1],
        -normal,
        config["ik_tolerance_m"],
        aligned["locked_joints"],
        thorough=True,
    )
    tip, wrist, axis = tool_geometry(arm, q)
    from .table_collision import clearances

    body, clearance = min(
        clearances(arm, q, {**config, "grasp_contact_mode": True}),
        key=lambda item: item[1],
    )
    entry = dict(
        grasp_preview=True,
        grasp_pose_rad=q.tolist(),
        table_normal=normal.tolist(),
        contact_m=contact.tolist(),
        pregrasp_m=above.tolist(),
        wrist_m=wrist.tolist(),
        tool_axis=axis.tolist(),
        tilt_deg=tilt_degrees(arm, q, normal),
        wrist_tip_length_m=float(np.linalg.norm(tip - wrist)),
        table_clearance_m=clearance,
        closest_table_body=body,
    )
    plan.planner_info = [
        i for i in plan.planner_info or [] if not i.get("grasp_preview")
    ] + [entry]
    return plan
