"""Single upright-can pick/lift/drop, with explicit preview and operator execution.

The two-view map locates the target. Nominal URDF jaw geometry estimates the
opening-dependent jaw center. It is not a contact-success-trained policy.
"""

import copy
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .grasp_geometry import tool_geometry
from .planning import Plan, check_workspace
from .table_collision import bodies
from .test_bed import table_height_at


def zone_points(points, z=0.0):
    import cv2

    p = np.asarray(points, float)
    if p.shape not in ((4, 2), (4, 3)) or not np.isfinite(p).all():
        raise ValueError("Select four finite discard-zone corners")
    xy = p[:, :2].astype(np.float32)
    if not cv2.isContourConvex(xy) or abs(cv2.contourArea(xy)) < 0.001:
        raise ValueError(
            "Discard zone must be a non-crossing quadrilateral at least 10 cm²"
        )
    return np.c_[xy, np.full(4, z)]


def inside_zone(point, zone):
    import cv2

    return (
        cv2.pointPolygonTest(
            np.ascontiguousarray(np.float32(zone)[:, :2]),
            tuple(map(float, np.asarray(point)[:2])),
            False,
        )
        >= 0
    )


# A 355 ml "211/202" can: 66 mm body, but the top necks down to a 202 end
# (~54 mm across the seamed lid). The lip a rim grasp pinches is at the lid
# radius, ~6 mm inside the body wall (manufacturer specs, not measured here).
LID_DIAMETER_M = 0.0541
NECK_HEIGHT_M = 0.019


def lid_radius(can):
    return float(can.get("lid_diameter_m", LID_DIAMETER_M)) / 2


def to_lid_radius(offset_m, can):
    """Move a body-radius rim proposal radially onto the real lid seam."""
    offset = np.asarray(offset_m, float).copy()
    radial = np.linalg.norm(offset[:2])
    if radial < 1e-6:
        return offset, 0.0
    shift = float(can["diameter_m"]) / 2 - lid_radius(can)
    offset[:2] *= max(radial - shift, 1e-6) / radial
    return offset, shift


HOLE_TIP_DEPTH_M = 0.010
# Gripper percent above the close command that indicates an object between
# the jaws after lifting (rim/lid pinch ~3 mm still stalls a few percent open).
GRIPPER_HOLD_MARGIN_PERCENT = 2.0
# The jaw must have risen at least this far since closing, with the can still
# detected in place on the table, to call an attempt a confirmed miss.
MISS_GAP_M = 0.015
HOLE_OUTSIDE_CLEARANCE_M = 0.004
HOLE_EXTRA_OPENING_M = 0.006


def hole_proposal(can, opening):
    """Jaw pose putting one tip down the pour hole and one outside the wall.

    ``opening`` comes from trajectory.can_opening.detect_opening. The closing
    axis is radial through the hole; the jaw opens wider than the hole-to-wall
    span (HOLE_EXTRA_OPENING_M) and then closes, pinching the lid edge.
    """
    lid = np.asarray(opening["lid_center_m"][:2], float)
    hole = np.asarray(opening["center_m"][:2], float)
    radial = hole - lid
    if np.linalg.norm(radial) < 0.005:
        raise ValueError("Pour hole is too close to the lid centre to set a direction")
    direction = radial / np.linalg.norm(radial)
    outside = lid + direction * (
        float(can["diameter_m"]) / 2 + HOLE_OUTSIDE_CLEARANCE_M
    )
    span = float(np.linalg.norm(outside - hole))
    middle = (outside + hole) / 2
    base = np.asarray(can["center_m"], float)
    yaw = float(np.arctan2(direction[1], direction[0]))
    return dict(
        grasp_mode="hole",
        offset_m=[
            float(middle[0] - base[0]),
            float(middle[1] - base[1]),
            float(can["height_m"]) - HOLE_TIP_DEPTH_M,
        ],
        quaternion_wxyz=[0.0, float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0.0],
        opening_m=span + HOLE_EXTRA_OPENING_M,
        hole_center_m=list(map(float, opening["center_m"])),
        lid_center_m=list(map(float, opening["lid_center_m"])),
        hole_confidence=float(opening["confidence"]),
        source="camera pour-hole detection; tips straddle the lid edge",
    )


def detect_can(boxes, config, depth=None, intrinsics=None, arm=None, q=None):
    candidates = []
    for box in boxes:
        size = box.high - box.low
        # A can-sized foreground instance under the user's single-can assumption.
        # Labels are not enough to infer object geometry; reject paper/noisy slivers.
        if (
            0.035 <= size[2] <= 0.18
            and 0.025 <= max(size[:2]) <= 0.115
            and min(size[:2]) > 0.015
        ):
            candidates.append(box)
    if not candidates:
        raise ValueError("No upright can-sized object confirmed in both views")
    box = max(candidates, key=lambda b: (b.confidence or 0, b.pixels))
    center = (box.low + box.high) / 2
    center[2] = float(table_height_at(center[:2], config)) - (
        config.get("test_bed") or {}
    ).get("clearance_margin_m", 0.003 if config.get("test_bed") else 0)
    source = "two-view RGB; assumed 66 mm diameter and 122.22 mm height"
    residual_mm = None
    if depth is not None and intrinsics is not None:
        from .trajectory.multiview_evidence import raised_cloud
        from .trajectory.yolo_perception import robot_image_mask

        mask = (
            robot_image_mask((*depth.shape, 3), config, intrinsics, arm, q)
            if arm is not None and q is not None
            else None
        )
        cloud, _ = raised_cloud(depth, intrinsics, config, mask)
        local = cloud[
            (np.linalg.norm(cloud[:, :2] - center[:2], axis=1) < 0.085)
            & (cloud[:, 2] > center[2] + 0.025)
            & (cloud[:, 2] < center[2] + 0.10)
        ]
        if len(local) >= 30:
            xy = local[:, :2]
            fit = least_squares(
                lambda c: np.linalg.norm(xy - c, axis=1) - 0.033,
                center[:2],
                bounds=(center[:2] - 0.05, center[:2] + 0.05),
                loss="soft_l1",
                f_scale=0.003,
            )
            residual = np.abs(np.linalg.norm(xy - fit.x, axis=1) - 0.033)
            if np.median(residual) < 0.004 and np.quantile(residual, 0.8) < 0.008:
                center[:2] = fit.x
                residual_mm = float(np.median(residual) * 1000)
                source = "two-view foreground + fixed-radius depth-side cylinder fit"
    return dict(
        center_m=center.tolist(),
        diameter_m=0.066,
        height_m=0.12222,
        lid_diameter_m=LID_DIAMETER_M,
        neck_height_m=NECK_HEIGHT_M,
        source=source,
        cylinder_fit_median_error_mm=residual_mm,
        confidence=box.confidence,
    )


def jaw_geometry(arm, percent):
    """Nominal moving fingertip relative to the calibrated closed-jaw contact."""
    fixed = arm.chain[-1][1]
    point = fixed[:3, 3] + fixed[:3, :3] @ arm.tool_offset
    jaw = bodies()[2]
    closed = np.linalg.inv(jaw) @ np.r_[point, 1]
    # Commissioned 0–100 opening is approximated by 0–100 URDF degrees.
    moving = (
        jaw[:3, :3]
        @ Rotation.from_rotvec([0, 0, np.deg2rad(percent)]).apply(closed[:3])
        + jaw[:3, 3]
    )
    delta = moving - point
    axis = point / np.linalg.norm(point)
    across = delta - axis * (delta @ axis)
    return delta / 2, float(np.linalg.norm(across))


def opening_for_width(arm, width):
    choices = np.linspace(0, 100, 401)
    gaps = np.array([jaw_geometry(arm, p)[1] for p in choices])
    eligible = np.flatnonzero(gaps >= width)
    if not len(eligible):
        raise ValueError("Can exceeds nominal gripper opening")
    return float(choices[eligible[0]])


def jaw_center(arm, q, percent):
    tip = arm.fk(q)
    # FK ends in the fixed tool frame; recover gripper-link orientation.
    frame = arm.frames(q)[1]
    rotation = frame[:3, :3] @ arm.chain[-1][1][:3, :3].T
    return tip + rotation @ jaw_geometry(arm, percent)[0]


def approach_tilt_limit(height, contact_height, contact_limit):
    """Allow approach tilt aloft, tightening continuously before contact."""
    clearance = max(0.0, float(height - contact_height))
    fraction = np.clip((clearance - 0.005) / 0.035, 0.0, 1.0)
    fraction = fraction * fraction * (3 - 2 * fraction)
    return float(contact_limit + (max(45.0, contact_limit) - contact_limit) * fraction)


def center_ik(arm, center, seed, percent, preferred_yaw=None, max_tilt_deg=20):
    axis_weight = 0.08

    def residual(q):
        axis = tool_geometry(arm, q)[2]
        closing_error = np.array([])
        if preferred_yaw is not None:
            t = arm.frames(q)[1]
            rotation = t[:3, :3] @ arm.chain[-1][1][:3, :3].T
            closing = (rotation @ jaw_geometry(arm, percent)[0])[:2]
            closing /= max(np.linalg.norm(closing), 1e-9)
            desired = np.array([np.cos(preferred_yaw), np.sin(preferred_yaw)])
            if closing @ desired < 0:
                desired = -desired
            closing_error = 0.015 * (closing - desired)
        return np.r_[
            jaw_center(arm, q, percent) - center,
            axis_weight * (axis - [0, 0, -1]),
            0.00005 * (q - seed),
            closing_error,
        ]

    best = None
    closest = None
    # Prioritize a reachable jaw-center position rather than allowing the
    # orientation penalty to displace it by centimeters near joint limits.
    for axis_weight in (0.08, 0.02, 0.005):
        for roll in (seed[4], 0.0, np.pi / 2, -np.pi / 2):
            initial = np.clip(seed, arm.lower + 1e-7, arm.upper - 1e-7).copy()
            initial[4] = np.clip(roll, arm.lower[4] + 1e-7, arm.upper[4] - 1e-7)
            fit = least_squares(
                residual, initial, bounds=(arm.lower, arm.upper), max_nfev=100
            )
            q = fit.x
            gap = np.linalg.norm(jaw_center(arm, q, percent) - center)
            tilt = np.arccos(np.clip(tool_geometry(arm, q)[2] @ [0, 0, -1], -1, 1))
            yaw_error = np.linalg.norm(residual(q)[11:])
            if gap < 0.002 and tilt < np.deg2rad(max_tilt_deg) and yaw_error < 0.004:
                score = gap + tilt * 0.02 + np.linalg.norm(q - seed) * 0.0001
                if best is None or score < best[0]:
                    best = (score, q)
                if (
                    gap < 0.0002
                    and tilt < np.deg2rad(max_tilt_deg)
                    and yaw_error < 0.0005
                ):
                    return q
        if best is not None:
            break

    if best is None:
        # Roll-only retries cannot escape a shoulder/elbow posture branch.
        # Keep the same physical acceptance tests; broaden only the search.
        alternatives = np.random.default_rng(101).uniform(arm.lower, arm.upper, (20, 5))
        for initial in alternatives:
            for axis_weight in (0.08, 0.02):
                fit = least_squares(
                    residual,
                    initial,
                    bounds=(arm.lower, arm.upper),
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=160,
                )
                q = fit.x
                gap = float(np.linalg.norm(jaw_center(arm, q, percent) - center))
                tilt = float(
                    np.arccos(np.clip(tool_geometry(arm, q)[2] @ [0, 0, -1], -1, 1))
                )
                yaw_error = float(np.linalg.norm(residual(q)[11:]))
                score = (
                    gap + 0.02 * max(0.0, tilt - np.deg2rad(max_tilt_deg)) + yaw_error
                )
                if closest is None or score < closest[0]:
                    closest = (score, gap, tilt, yaw_error)
                if (
                    gap < 0.002
                    and tilt < np.deg2rad(max_tilt_deg)
                    and yaw_error < 0.004
                ):
                    return q
        _, gap, tilt, yaw_error = closest
        raise ValueError(
            f"Downward IK: best position error {gap * 1000:.1f} mm (max 2), "
            f"tilt {np.rad2deg(tilt):.1f} deg (max {max_tilt_deg}), "
            f"yaw residual {yaw_error:.4f} (max .004)"
        )
    return best[1]


def carried_can_clearance(arm, poses, grasp_pose, can, closed, config):
    """Nominal rigid attachment: include the held cylinder against the table."""
    from .table_collision import physical_plane

    angle = np.arange(24) * 2 * np.pi / 24
    center = np.asarray(can["center_m"])
    ring = np.c_[np.cos(angle), np.sin(angle)] * can["diameter_m"] / 2 + center[:2]
    vertices = np.vstack(
        [np.c_[ring, np.full(24, center[2] + z)] for z in (0, can["height_m"])]
    )

    def rotation(q):
        return arm.frames(q)[1][:3, :3] @ arm.chain[-1][1][:3, :3].T

    relative = (vertices - jaw_center(arm, grasp_pose, closed)) @ rotation(grasp_pose)
    normal, offset = physical_plane(config)
    return min(
        float(
            np.min(
                (relative @ rotation(q).T + jaw_center(arm, q, closed)) @ normal
                + offset
            )
        )
        for q in poses
    )


def pack(plan):
    return dict(
        time_s=plan.time_s.tolist(),
        joints_rad=plan.joints_rad.tolist(),
        target_m=plan.target_m.tolist(),
        planner_info=plan.planner_info,
    )


def unpack(value):
    return Plan(
        np.array(value["time_s"]),
        np.array(value["joints_rad"]),
        np.array(value["target_m"]),
        0.0,
        planner_info=value["planner_info"],
    )


def plan_can_cycle(arm, start, config, can, proposal=None, deadline=None):
    from .trajectory.joint_planner import JointSpace, plan_joint_route, timed_joint_plan

    cfg = copy.deepcopy(config)
    cfg.update(
        locked_joints=[],
        allow_object_collisions=False,
        click_and_grab=False,
        grasp_request=None,
    )
    zone = zone_points(config.get("discard_zone"), config["table_z_m"])
    center = np.asarray(can["center_m"]).copy()
    yaw = None
    rim = proposal is not None and proposal.get("grasp_mode") == "rim"
    hole = proposal is not None and proposal.get("grasp_mode") == "hole"
    opening = can["diameter_m"] + 0.006
    if proposal is not None:
        center += np.array(proposal["offset_m"])
        w, x, y, z = proposal["quaternion_wxyz"]
        yaw = float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))
        opening = (
            float(np.clip(proposal["opening_m"], 0.005, 0.012))
            if rim
            else float(np.clip(proposal["opening_m"], 0.012, 0.045))
            if hole
            else float(
                np.clip(
                    proposal["opening_m"],
                    can["diameter_m"] + 0.002,
                    can["diameter_m"] + 0.012,
                )
            )
        )
    else:
        center[2] += 0.55 * can["height_m"]
    opened = opening_for_width(arm, opening)
    # Command a close well past the surface: a position servo only squeezes in
    # proportion to how far its goal is beyond contact (torque-limited).
    closed = opening_for_width(
        arm, 0.0015 if rim or hole else max(0.0015, can["diameter_m"] - 0.012)
    )
    # Where the jaws actually stop on the can; used for held-can geometry.
    contact = opening_for_width(
        arm, 0.0015 if rim or hole else can["diameter_m"] - 0.001
    )
    # The hole tip descends vertically through a ~14 mm opening; 25 degrees of
    # jaw tilt shifts the finger shaft ~4.7 mm over a 10 mm insertion.
    max_tilt = 35 if rim else 25 if hole else 20
    q = center_ik(
        arm, center, np.asarray(start), opened, preferred_yaw=yaw, max_tilt_deg=max_tilt
    )
    # Respect operator Z bounds. Raising the grasp is allowed only while still
    # on the can body; never lower the floor to make it feasible.
    for _ in range(3):
        deficit = cfg["workspace_min_m"][2] - arm.fk(q)[2]
        if deficit <= 0:
            break
        if rim or hole:
            raise ValueError(
                "Workspace Z floor excludes the can lip; no descent planned"
            )
        center[2] += deficit + 0.002
        if center[2] > can["center_m"][2] + 0.85 * can["height_m"]:
            raise ValueError(
                "Z floor is too high to grasp the can body; no descent planned"
            )
        q = center_ik(arm, center, q, opened, preferred_yaw=yaw, max_tilt_deg=max_tilt)
    check_workspace(arm.fk(q), cfg, label="Can grasp tip")
    above_q = None
    pregrasp_failures = []
    for lift in sorted(
        {max(0.02, min(0.05, config["hover_m"])), 0.04, 0.03, 0.02}, reverse=True
    ):
        above = center + [0, 0, lift]
        try:
            above_q = center_ik(
                arm,
                above,
                q,
                opened,
                preferred_yaw=yaw,
                max_tilt_deg=approach_tilt_limit(above[2], center[2], max_tilt),
            )
            check_workspace(arm.fk(above_q), cfg, label="Can pregrasp tip")
            break
        except ValueError as exc:
            pregrasp_failures.append(f"+{lift * 100:.0f} cm: {exc}")
            above_q = None
    if above_q is None:
        raise ValueError("Can pregrasp failed - " + "; ".join(pregrasp_failures))
    first = plan_joint_route(
        arm, start, arm.fk(above_q), cfg, [], 1, 0, goal_pose=above_q
    )

    def vertical(a, b, seed):
        path = [seed]
        for t in np.linspace(0, 1, 13)[1:]:
            path.append(
                center_ik(
                    arm,
                    a * (1 - t) + b * t,
                    path[-1],
                    opened,
                    preferred_yaw=yaw,
                    max_tilt_deg=approach_tilt_limit(
                        (a * (1 - t) + b * t)[2], center[2], max_tilt
                    ),
                )
            )
        space = JointSpace(arm, cfg, [], path[0])
        if not space.path_free(path):
            raise ValueError(
                "Can vertical approach conflicts with table, self or bounds"
            )
        return timed_joint_plan(
            arm,
            path,
            arm.fk(path[-1]),
            cfg,
            dict(
                trajectory_method="can vertical",
                obstacles_m=[],
                scene_revision=0,
                allow_object_collisions=False,
            ),
        )

    down = vertical(above, center, above_q)
    up = vertical(center, above, down.joints_rad[-1])
    if (
        carried_can_clearance(
            arm, up.joints_rad, down.joints_rad[-1], can, contact, cfg
        )
        < -0.001
    ):
        raise ValueError("Held can would overlap the table during lift")
    centroid = zone[:, :2].mean(0)
    preferred_z = min(
        cfg["workspace_max_m"][2] - 0.005,
        max(cfg["workspace_min_m"][2], arm.fk(above_q)[2]),
    )
    transport = None
    deadline = (
        min(time.monotonic() + 15, deadline)
        if deadline is not None
        else time.monotonic() + 15
    )
    # A valid zone may have only a reachable edge/corner. Sample the actual
    # quadrilateral rather than only its central 35 percent.
    samples = [centroid]
    for corner in zone:
        samples.append(corner[:2] * 0.999 + centroid * 0.001)
    for a, b in zip(zone, np.roll(zone, -1, axis=0)):
        for fraction in (0.25, 0.5, 0.75):
            samples.append(
                (a[:2] * (1 - fraction) + b[:2] * fraction) * 0.999 + centroid * 0.001
            )

    def drop_cost(xy):
        pan = -np.arctan2(xy[1], xy[0])
        violation = max(arm.lower[0] - pan, pan - arm.upper[0], 0)
        return violation * 1000 + np.linalg.norm(xy)

    samples.sort(key=drop_cost)
    for xy in samples:
        for height in [
            preferred_z,
            max(cfg["workspace_min_m"][2] + 0.002, preferred_z - 0.02),
            min(cfg["workspace_max_m"][2] - 0.002, preferred_z + 0.02),
        ]:
            if time.monotonic() > deadline:
                break
            drop = np.r_[xy, height]
            if not inside_zone(drop, zone):
                continue
            try:
                # The discard goal constrains the tip position, not a new
                # vertical grasp orientation. Preserve the lifted pose as the
                # IK seed, then validate the attached can along the whole route.
                drop_q = arm.ik(drop, up.joints_rad[-1], cfg["ik_tolerance_m"], [])
                candidate = plan_joint_route(
                    arm, up.joints_rad[-1], drop, cfg, [], 1, 0, goal_pose=drop_q
                )
                if (
                    carried_can_clearance(
                        arm,
                        candidate.joints_rad,
                        down.joints_rad[-1],
                        can,
                        contact,
                        cfg,
                    )
                    < -0.001
                ):
                    continue
                transport = candidate
                break
            except ValueError:
                pass
        if transport is not None or time.monotonic() > deadline:
            break
    if transport is None:
        raise ValueError(
            "No validated drop route found in the selected discard zone; choose a closer zone"
        )
    phases = [first, down, up, transport]
    times = [0.0]
    poses = [first.joints_rad[0]]
    for phase in phases:
        times.extend((phase.time_s[1:] + times[-1]).tolist())
        poses.extend(phase.joints_rad[1:])
    request = dict(
        can=can,
        grasp_method="camera pour-hole"
        if hole
        else "imf"
        if proposal is not None
        else "numerical cylinder baseline",
        model_proposal=proposal,
        phases=[pack(p) for p in phases],
        open_percent=opened,
        close_percent=closed,
        contact_percent=contact,
        drop_tip_m=drop.tolist(),
        jaw_mapping="nominal URDF; not measured force/contact",
        verification="not yet physically validated",
        maximum_approach_tilt_deg=max_tilt,
        grasp_center_m=center.tolist(),
    )
    combined = Plan(
        np.array(times),
        np.array(poses),
        drop,
        0.0,
        planner_info=[dict(can_transfer=request)],
    )
    cfg["can_request"] = request
    return combined, cfg


def learned_can_cycle(arm, start, config, can, excluded_angles=()):
    from .learned.grasp_proposals import propose

    rows = propose(
        [can["diameter_m"], can["diameter_m"], can["height_m"]],
        candidates=32,
        checkpoint="runs/deskbuddy/can-rim-grasps-v1/imf.pt",
    )
    rows = [row for row in rows if row["geometric_valid"]]
    # The model was trained with the lip at the body radius; the real lip is
    # on the narrower lid. Correct each proposal before planning.
    for row in rows:
        offset, shift = to_lid_radius(row["offset_m"], can)
        row["offset_m"] = offset.tolist()
        row["lid_radius_correction_m"] = shift
    rows = [
        row
        for row in rows
        if all(
            abs(
                (np.arctan2(row["offset_m"][1], row["offset_m"][0]) - angle + np.pi)
                % (2 * np.pi)
                - np.pi
            )
            > np.deg2rad(30)
            for angle in excluded_angles
        )
    ]
    # Prefer the lip facing the current jaw, then try other sampled approaches.
    current_xy = jaw_center(arm, start, 0)[:2] - np.asarray(can["center_m"])[:2]
    current_xy /= max(np.linalg.norm(current_xy), 1e-9)
    rows.sort(key=lambda row: -np.asarray(row["offset_m"][:2]) @ current_xy)
    feasible = []
    reasons = []
    deadline = time.monotonic() + 20
    for row in rows[:3]:
        if time.monotonic() >= deadline:
            break
        try:
            plan, cfg = plan_can_cycle(
                arm, start, config, can, proposal=row, deadline=deadline
            )
            feasible.append((plan, cfg))
            break  # First fully validated proposal; avoid three redundant whole-cycle plans.
        except ValueError as exc:
            reasons.append(str(exc))
    if not feasible:
        raise ValueError(
            "iMF: no executable can grasp; "
            + (reasons[-1] if reasons else "no geometrically valid candidates")
        )
    plan, cfg = min(feasible, key=lambda item: item[0].time_s[-1])
    cfg["can_request"]["grasp_mode"] = "imf_rim"
    cfg["can_request"]["candidates_sampled"] = 32
    cfg["can_request"]["executable_candidates_checked"] = len(feasible)
    return plan, cfg


def fingertips(arm, q, percent):
    """Fixed and moving fingertip positions at a jaw opening (nominal URDF)."""
    fixed = arm.fk(q)
    return fixed, 2 * jaw_center(arm, q, percent) - fixed


def center_tip_in_hole(arm, start, can, proposal, iterations=4):
    """Shift the jaw so the in-hole fingertip is dead centre on the hole.

    The jaw centre is the midpoint of two asymmetric fingers (one fixed, one
    rotating), so placing the midpoint geometrically left the inside tip a few
    millimetres off the hole centre, i.e. near its edge. Solve the actual
    grasp pose, measure where each tip lands, and move the jaw until the tip
    nearest the hole sits on its centre at the insertion depth. The opening is
    widened if the outside tip would not clear the can wall.
    """
    proposal = dict(proposal)
    base = np.asarray(can["center_m"], float)
    hole = np.asarray(proposal["hole_center_m"], float)
    target_z = base[2] + float(can["height_m"]) - HOLE_TIP_DEPTH_M
    w, x, y, z = proposal["quaternion_wxyz"]
    yaw = float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))
    wall = float(can["diameter_m"]) / 2 + 0.002
    inside = None
    for _ in range(iterations):
        opening = float(np.clip(proposal["opening_m"], 0.012, 0.045))
        percent = opening_for_width(arm, opening)
        center = base + np.asarray(proposal["offset_m"])
        q = center_ik(arm, center, start, percent, preferred_yaw=yaw, max_tilt_deg=25)
        tips = fingertips(arm, q, percent)
        index = int(np.argmin([np.linalg.norm(t[:2] - hole[:2]) for t in tips]))
        inside, outside = tips[index], tips[1 - index]
        error = np.r_[hole[:2] - inside[:2], target_z - inside[2]]
        clearance = wall - np.linalg.norm(outside[:2] - base[:2])
        offset = np.asarray(proposal["offset_m"], float) + error
        proposal["offset_m"] = offset.tolist()
        if clearance > 0:
            proposal["opening_m"] = opening + clearance + 0.001
        if np.linalg.norm(error) < 0.0005 and clearance <= 0:
            break
    proposal.update(
        inside_tip="fixed" if index == 0 else "moving",
        planned_tip_error_m=float(np.linalg.norm(hole[:2] - inside[:2])),
    )
    return proposal


def hole_can_cycle(arm, start, config, can, opening):
    """Plan the pour-hole grasp; the can axis comes from the edge-fitted lid."""
    can = dict(can)
    center = np.asarray(can["center_m"], float).copy()
    center[:2] = opening["lid_center_m"][:2]
    can["center_m"] = center.tolist()
    can["source"] = str(can.get("source", "")) + " + edge-fitted lid centre"
    proposal = center_tip_in_hole(arm, start, can, hole_proposal(can, opening))
    plan, cfg = plan_can_cycle(arm, start, config, can, proposal=proposal)
    cfg["can_request"]["grasp_mode"] = "hole"
    cfg["can_request"]["opening_detection"] = {
        k: v for k, v in opening.items() if k not in ("pixel",)
    }
    return plan, cfg


def run_transfer(move, grip, stopped, opened, closed):
    for operation, value in [
        ("grip", opened),
        ("move", 0),
        ("move", 1),
        ("grip", closed),
        ("move", 2),
        ("move", 3),
        ("grip", opened),
    ]:
        if stopped():
            raise RuntimeError("Can transfer stopped; current hold retained")
        (move if operation == "move" else grip)(value)


def checked_transfer(prepare, move, grip, verify, stopped, publish, max_attempts=5):
    """Only retry confirmed misses; uncertain evidence never opens a held can."""
    for attempt in range(max_attempts):
        if stopped():
            raise RuntimeError("Can sequence stopped; position hold retained")
        opened, closed = prepare(attempt)
        publish(
            dict(
                can_attempt=attempt + 1, phase=f"Can grasp {attempt + 1}/{max_attempts}"
            )
        )
        for action, value in (
            (grip, opened),
            (move, 0),
            (move, 1),
            (grip, closed),
            (move, 2),
        ):
            if stopped():
                raise RuntimeError("Can sequence stopped; position hold retained")
            action(value)
        followed = verify()
        if stopped():
            raise RuntimeError("Can sequence stopped; position hold retained")
        if followed is None:
            raise RuntimeError(
                "Pickup uncertain in RGB; retaining hold without transfer or retry"
            )
        if followed:
            move(3)
            if stopped():
                raise RuntimeError("Can sequence stopped before release; hold retained")
            grip(opened)
            return dict(
                completed=True,
                can_discard_completed=True,
                pickup_verified=True,
                pickup_verification="RGB motion estimate, not force sensing",
                can_attempts=attempt + 1,
            )
        publish(
            dict(
                can_attempt=attempt + 1,
                phase="Can stayed on table; trying another approach"
                if attempt + 1 < max_attempts
                else "Can pickup failed after five attempts",
            )
        )
    return dict(
        completed=False,
        can_discard_completed=False,
        pickup_verified=False,
        can_attempts=max_attempts,
        note="Can stayed on table; attempts exhausted, position hold retained",
    )


def execute_can(bus, selected, config, stopped, publish, read_live=None):
    from .can_pickup import PickupVerifier
    from .execute_reach import execute_target
    from .gripper_control import ramp_gripper
    from .kinematics import JOINTS, Arm

    request = config["can_request"]
    phases = [unpack(p) for p in request["phases"]]
    last = [selected, {}]
    cfg = copy.deepcopy(config)
    cfg.update(
        can_request=None,
        grasp_request=None,
        click_and_grab=False,
        allow_object_collisions=False,
        locked_joints=[],
    )
    arm = Arm(cfg.get("model_joint_limits_deg"), cfg.get("tool_offset_m"))
    excluded = []
    verifier = None
    grasp_pose = None

    def observed():
        from .execute_reach import model_pose

        return model_pose(
            bus.sync_read("Present_Position", list(JOINTS), num_retry=2), cfg
        )

    def prepare(attempt):
        nonlocal phases, request, verifier
        if attempt:
            if request.get("grasp_mode") == "imf_rim":
                _, job = learned_can_cycle(
                    arm, observed(), cfg, request["can"], excluded_angles=excluded
                )
            else:
                yaw = attempt * np.pi / 3
                can = request["can"]
                proposal = dict(
                    grasp_mode="body",
                    offset_m=[0, 0, can["height_m"] * 0.55],
                    quaternion_wxyz=[0, np.cos(yaw / 2), np.sin(yaw / 2), 0],
                    opening_m=can["diameter_m"] + 0.006,
                    source="numerical retry yaw",
                )
                _, job = plan_can_cycle(arm, observed(), cfg, can, proposal=proposal)
                job["can_request"]["grasp_method"] = "numerical body retry"
            request = job["can_request"]
            phases = [unpack(p) for p in request["phases"]]
        proposal = request.get("model_proposal")
        if request.get("grasp_mode") == "imf_rim" and proposal:
            excluded.append(
                float(np.arctan2(proposal["offset_m"][1], proposal["offset_m"][0]))
            )
        # Capture after planning so baseline frames cannot be stale on arrival.
        verifier = PickupVerifier(cfg, request["can"], read_live, stopped)
        return request["open_percent"], request["close_percent"]

    def can_left_behind(lifted_at, frames=2, timeout=3.0):
        """Object detection after the lift: is the can still on the table,
        well below the jaw? True = confirmed miss, False = not there, None =
        no fresh detection.
        """
        if read_live is None:
            return None
        original = np.asarray(request["can"]["center_m"][:2], float)
        contact = request.get("contact_percent", request["close_percent"])
        # How far the jaw rose since closing: a held can's bottom would now be
        # this far above the table, so a can still detected on the table with
        # this gap is clearly not in the gripper.
        gap = float(
            jaw_center(arm, observed(), contact)[2]
            - jaw_center(arm, grasp_pose, contact)[2]
        )
        seen, last_stamp = [], lifted_at + 0.2
        deadline = time.monotonic() + timeout
        while len(seen) < frames and time.monotonic() < deadline and not stopped():
            try:
                detection = (read_live() or {}).get("obstacle_detection") or {}
            except (OSError, ValueError):
                detection = {}
            stamp = detection.get("detection_unix_s", 0)
            if stamp > last_stamp:
                last_stamp = stamp
                on_table = False
                for box in detection.get("raw_obstacle_boxes", []):
                    low, high = np.asarray(box["low_m"]), np.asarray(box["high_m"])
                    if (
                        np.linalg.norm((low[:2] + high[:2]) / 2 - original) < 0.025
                        and 0.08 < high[2] - low[2] < 0.16
                    ):
                        on_table = True
                seen.append(on_table)
            else:
                time.sleep(0.1)
        publish(
            dict(
                pickup_detection=dict(
                    frames=len(seen),
                    can_on_table=seen,
                    jaw_lift_mm=round(gap * 1000, 1),
                )
            )
        )
        if len(seen) < frames:
            return None
        if all(seen) and gap >= MISS_GAP_M:
            return True
        if not any(seen):
            return False
        return None

    def verify_pickup():
        lifted_at = time.time()
        followed = verifier.check(
            arm,
            grasp_pose,
            observed(),
            request.get("contact_percent", request["close_percent"]),
        )
        if followed:
            return True
        # Before calling it a miss (or giving up as uncertain), confirm with
        # object detection: the can still on the table, a clear gap below the
        # lifted jaw. Only a confirmed miss triggers a new attempt.
        left = can_left_behind(lifted_at)
        if left:
            publish(dict(phase="Can still on the table; retrying"))
            return False
        if followed is False and left is None:
            return False
        # Cameras undecided: a gripper held open well past its close command
        # after the lift means something is still between the jaws.
        try:
            position = float(
                bus.sync_read("Present_Position", ["gripper"], num_retry=2)["gripper"]
            )
        except (OSError, KeyError, RuntimeError, ValueError):
            return None
        if position >= request["close_percent"] + GRIPPER_HOLD_MARGIN_PERCENT:
            publish(
                dict(
                    pickup_evidence=f"gripper stalled at {position:.1f}% "
                    f"(commanded {request['close_percent']:.1f}%) after lift",
                )
            )
            return True
        return None

    def run_joints(joints, info):
        from .trajectory.joint_planner import timed_joint_plan

        plan = timed_joint_plan(
            arm,
            np.vstack([observed(), np.asarray(joints)]),
            arm.fk(np.asarray(joints)[-1]),
            cfg,
            info,
        )
        last[:] = execute_target(
            bus, plan.target_m, cfg, stopped, publish, selected_plan=plan
        )

    def fresh_measurement():
        from .can_pickup import read_images
        from .can_servo import measure, save_log

        arrived = time.time()
        frames = None
        deadline = time.monotonic() + 2.0
        while frames is None and time.monotonic() < deadline and not stopped():
            try:
                frames = read_images(read_live, arrived)
            except (OSError, ValueError):
                frames = None
            if frames is None:
                time.sleep(0.05)
        if frames is None:
            return None
        _, images, live = frames
        debug = []
        try:
            report = measure(
                images,
                live["intrinsics"],
                cfg,
                arm,
                observed(),
                request["open_percent"],
                request["can"],
                debug,
            )
        except (ValueError, KeyError, RuntimeError):
            return None
        try:
            report["log_folder"] = str(save_log(report, images, debug))
        except OSError:
            pass
        return report

    def hole_miss():
        """Camera estimate of in-hole tip minus hole centre (XY), or None."""
        report = fresh_measurement()
        if report is None or "relative_error_m" not in report:
            return None
        proposal = request["model_proposal"]
        fixed, moving = fingertips(arm, observed(), request["open_percent"])
        tip = fixed if proposal.get("inside_tip") == "fixed" else moving
        return (
            tip[:2]
            + np.asarray(report["relative_error_m"])
            - np.asarray(proposal["hole_center_m"][:2])
        )

    def shift_and_replan(miss, retract):
        """Back out along the path we came, then replan aimed at the hole."""
        nonlocal phases, request
        if len(retract):
            run_joints(
                retract,
                dict(
                    trajectory_method="hole retract",
                    obstacles_m=[],
                    scene_revision=0,
                    allow_object_collisions=False,
                ),
            )
        can = dict(request["can"])
        center = np.asarray(can["center_m"], float).copy()
        center[:2] -= miss
        can["center_m"] = center.tolist()
        proposal = dict(request["model_proposal"])
        proposal["hole_center_m"] = (
            np.asarray(proposal["hole_center_m"]) - np.r_[miss, 0]
        ).tolist()
        _, job = plan_can_cycle(arm, observed(), cfg, can, proposal=proposal)
        updated = job["can_request"]
        for key in ("grasp_mode", "grasp_method", "opening_detection"):
            if key in request:
                updated[key] = request[key]
        updated.setdefault("hole_readjustments", [])
        updated["hole_readjustments"] = request.get("hole_readjustments", []) + [
            np.asarray(miss).tolist()
        ]
        request = updated
        phases = [unpack(p) for p in request["phases"]]
        hop = phases[0]
        run_joints(hop.joints_rad[1:], hop.planner_info[0])

    def hole_insertion(tolerance_entry=0.0025, tolerance_inserted=0.004, tries=3):
        """Descend to just above the hole, check with cameras, insert, re-check.

        A confirmed miss backs the jaw out along its own path and replans
        aimed at the measured hole; unknown evidence proceeds (the pickup
        check after lifting still catches a failed grasp).
        """
        for attempt in range(tries):
            down = phases[1].joints_rad
            seam_z = request["can"]["center_m"][2] + request["can"]["height_m"]
            inside = request["model_proposal"].get("inside_tip", "moving")
            heights = [
                fingertips(arm, q, request["open_percent"])[
                    0 if inside == "fixed" else 1
                ][2]
                for q in down
            ]
            entry = next(
                (i for i, h in enumerate(heights) if h <= seam_z + 0.004),
                len(down) - 1,
            )
            info = dict(
                trajectory_method="hole descent",
                obstacles_m=[],
                scene_revision=0,
                allow_object_collisions=False,
            )
            if entry >= 1:
                run_joints(down[1 : entry + 1], info)
            miss = hole_miss()
            publish(
                dict(
                    phase="Pour hole: aligning above opening",
                    hole_miss_mm=None
                    if miss is None
                    else float(np.linalg.norm(miss) * 1000),
                )
            )
            last_attempt = attempt == tries - 1
            if (
                miss is not None
                and np.linalg.norm(miss) > tolerance_entry
                and not last_attempt
            ):
                shift_and_replan(miss, down[:entry][::-1])
                continue
            if entry + 1 < len(down):
                run_joints(down[entry + 1 :], info)
            miss = hole_miss()
            publish(
                dict(
                    phase="Pour hole: checking tip is inside",
                    hole_miss_mm=None
                    if miss is None
                    else float(np.linalg.norm(miss) * 1000),
                )
            )
            if (
                miss is not None
                and np.linalg.norm(miss) > tolerance_inserted
                and not last_attempt
            ):
                shift_and_replan(miss, down[:-1][::-1])
                continue
            return

    def move(index):
        nonlocal grasp_pose
        from .trajectory.joint_planner import timed_joint_plan

        if (
            index == 1
            and read_live is not None
            and request.get("grasp_mode") == "hole"
            and request.get("model_proposal")
        ):
            hole_insertion()
            grasp_pose = observed()
            return
        plan = phases[index]
        plan = timed_joint_plan(
            arm,
            np.vstack([observed(), plan.joints_rad[1:]]),
            plan.target_m,
            cfg,
            plan.planner_info[0],
        )
        last[:] = execute_target(
            bus, plan.target_m, cfg, stopped, publish, selected_plan=plan
        )
        if index == 0 and read_live is not None:
            camera_guidance()
        if index == 1:
            grasp_pose = observed()

    def camera_guidance():
        """Close the webcam loop above the can, then rebuild the descent."""
        nonlocal phases, request
        import json
        from pathlib import Path

        from .local_visual_servo import measure_jacobian, refine
        from .trajectory.joint_planner import plan_joint_route
        from .webcam_servo import measurement, read_frame

        desired = jaw_center(arm, phases[1].joints_rad[-1], request["open_percent"])
        original_can = copy.deepcopy(request["can"])
        hole = request.get("grasp_mode") == "hole"
        last_frame = time.time()

        def observe_image():
            nonlocal last_frame
            deadline = time.monotonic() + 3.5
            samples = []
            while len(samples) < 3 and time.monotonic() < deadline and not stopped():
                try:
                    frames = read_frame(read_live, last_frame)
                    if frames is None:
                        time.sleep(0.04)
                        continue
                    stamp, image, tracked = frames
                    last_frame = stamp
                    if tracked and (
                        not tracked.get("valid")
                        or abs(stamp - tracked.get("unix_time_s", 0)) > 0.5
                    ):
                        continue
                    view_config = dict(cfg, can_request=request)
                    sample = measurement(
                        image,
                        view_config,
                        arm,
                        observed(),
                        request["open_percent"],
                        original_can,
                        desired,
                        hole=hole,
                        tracked_tip=tracked["pixel"] if tracked else None,
                    )
                    if sample is not None:
                        samples.append(sample)
                except (ValueError, KeyError, RuntimeError, OSError):
                    time.sleep(0.04)
            if len(samples) < 2:
                return None
            return dict(
                error_px=np.median([r["error_px"] for r in samples], axis=0),
                jacobian_px_m=np.median([r["jacobian_px_m"] for r in samples], axis=0),
            )

        def move_xy(delta):
            before = observed()
            position = jaw_center(arm, before, request["open_percent"])
            goal_center = position + np.r_[delta, 0.0]
            goal = center_ik(
                arm,
                goal_center,
                before,
                request["open_percent"],
                max_tilt_deg=approach_tilt_limit(
                    position[2],
                    desired[2],
                    35 if request.get("grasp_mode") == "rim" else 25 if hole else 20,
                ),
            )
            if np.max(np.abs(goal - before)) > np.deg2rad(5):
                raise ValueError(
                    "Visual correction requires a different posture; retaining the planned approach"
                )
            plan = plan_joint_route(
                arm, before, arm.fk(goal), cfg, [], 1, 0, goal_pose=goal
            )
            last[:] = execute_target(
                bus, plan.target_m, cfg, stopped, publish, selected_plan=plan
            )
            return (jaw_center(arm, observed(), request["open_percent"]) - position)[:2]

        try:
            current_tracking = (read_live() or {}).get("tip_tracking")
            if not current_tracking or not current_tracking.get("valid"):
                publish(
                    dict(
                        camera_guidance="Select tip in webcam before measured fine alignment; using planned approach"
                    )
                )
                return
            publish(dict(phase="Measuring webcam response with 2 mm XY probes"))
            probe = measure_jacobian(observe_image, move_xy, stopped)
            publish(dict(visual_probe=probe))
            if not probe["ready"]:
                if stopped():
                    raise RuntimeError("Operator stopped visual probes; hold retained")
                publish(dict(camera_guidance=probe["status"]))
                return
            report = refine(
                observe_image,
                move_xy,
                stopped,
                lambda row: publish(
                    dict(phase="Webcam fine alignment", visual_servo=row)
                ),
                initial_jacobian=probe["jacobian_px_m"],
            )
            report["probe"] = probe
        except (ValueError, RuntimeError) as exc:
            if stopped():
                raise
            raise ValueError(f"Webcam correction could not complete: {exc}") from exc
        publish(dict(camera_guidance=report["status"], visual_servo=report))
        if stopped():
            raise RuntimeError("Operator stopped visual alignment; hold retained")
        directory = Path("runs/servo-log")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"visual-servo-{time.time_ns()}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        correction = np.asarray(report["correction_m"])
        if np.linalg.norm(correction) < 0.0001:
            return
        # Carry the learned offset into the remaining descent, rather than
        # returning to the original (uncorrected) phase after visual alignment.
        can = copy.deepcopy(original_can)
        can["center_m"] = (
            np.asarray(can["center_m"]) + np.r_[correction, 0.0]
        ).tolist()
        proposal = copy.deepcopy(request.get("model_proposal"))
        if proposal and proposal.get("hole_center_m"):
            proposal["hole_center_m"] = (
                np.asarray(proposal["hole_center_m"]) + np.r_[correction, 0.0]
            ).tolist()
        _, job = plan_can_cycle(arm, observed(), cfg, can, proposal=proposal)
        updated = job["can_request"]
        for key in ("grasp_mode", "grasp_method", "opening_detection"):
            if key in request:
                updated[key] = request[key]
        updated["visual_servo"] = report
        request = updated
        phases = [unpack(p) for p in request["phases"]]
        hop = phases[0]
        run_joints(hop.joints_rad[1:], hop.planner_info[0])

    def grip(percent):
        ramp_gripper(bus, percent, stopped)

    outcome = checked_transfer(
        prepare,
        move,
        grip,
        verify_pickup,
        stopped,
        publish,
    )
    last[1].update(
        outcome,
        drop_tip_m=request["drop_tip_m"],
        gripper_unchanged=False,
        drop_verified=False,
    )
    return last
