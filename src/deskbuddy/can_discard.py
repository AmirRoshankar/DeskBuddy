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
        raise ValueError("No reachable downward can grasp at this location")
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
    opening = can["diameter_m"] + 0.006
    if proposal is not None:
        center += np.array(proposal["offset_m"])
        w, x, y, z = proposal["quaternion_wxyz"]
        yaw = float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))
        opening = (
            float(np.clip(proposal["opening_m"], 0.005, 0.012))
            if rim
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
    closed = opening_for_width(arm, 0.0015 if rim else can["diameter_m"] - 0.001)
    max_tilt = 35 if rim else 20
    q = center_ik(
        arm, center, np.asarray(start), opened, preferred_yaw=yaw, max_tilt_deg=max_tilt
    )
    # Respect operator Z bounds. Raising the grasp is allowed only while still
    # on the can body; never lower the floor to make it feasible.
    for _ in range(3):
        deficit = cfg["workspace_min_m"][2] - arm.fk(q)[2]
        if deficit <= 0:
            break
        if rim:
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
    for lift in sorted(
        {max(0.02, min(0.05, config["hover_m"])), 0.04, 0.03, 0.02}, reverse=True
    ):
        above = center + [0, 0, lift]
        try:
            above_q = center_ik(
                arm, above, q, opened, preferred_yaw=yaw, max_tilt_deg=max_tilt
            )
            check_workspace(arm.fk(above_q), cfg, label="Can pregrasp tip")
            break
        except ValueError:
            above_q = None
    if above_q is None:
        raise ValueError("No reachable pregrasp above the can within current bounds")
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
                    max_tilt_deg=max_tilt,
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
        carried_can_clearance(arm, up.joints_rad, down.joints_rad[-1], can, closed, cfg)
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
                        arm, candidate.joints_rad, down.joints_rad[-1], can, closed, cfg
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
        grasp_method="imf" if proposal is not None else "numerical cylinder baseline",
        model_proposal=proposal,
        phases=[pack(p) for p in phases],
        open_percent=opened,
        close_percent=closed,
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
        checkpoint="runs/flowreach/can-rim-grasps-v1/imf.pt",
    )
    rows = [row for row in rows if row["geometric_valid"]]
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

    def move(index):
        nonlocal grasp_pose
        from .trajectory.joint_planner import timed_joint_plan

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
        if index == 1:
            grasp_pose = observed()

    def grip(percent):
        ramp_gripper(bus, percent, stopped)

    outcome = checked_transfer(
        prepare,
        move,
        grip,
        lambda: verifier.check(arm, grasp_pose, observed(), request["close_percent"]),
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
