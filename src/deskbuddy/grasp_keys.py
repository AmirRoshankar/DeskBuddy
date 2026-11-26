"""Bounded open–approach–descend–close–lift–verify key-grasp state machine."""

import time

import numpy as np

from .kinematics import JOINTS, Arm
from .overlays import project
from .test_bed import surface_point, table_height_at


def detect_key_target(report, intrinsics, config):
    keys = [
        k
        for k in report.get("key_detections", [])
        if k["confidence"] >= 0.25 and k.get("confirmed_frames", 2) >= 2
    ]
    for key in sorted(keys, key=lambda k: k["confidence"], reverse=True):
        x1, y1, x2, y2 = key["pixel_roi"]
        try:
            pixel = key.get("grasp_pixel", [(x1 + x2) / 2, (y1 + y2) / 2])
            return surface_point(*pixel, intrinsics, config), key
        except ValueError:
            continue
    raise ValueError(
        "No confident keys on the paper. Click the key/chain and use Click & grab."
    )


def vision_grasped(live, source, hover, config, reference_pixel):
    report = live.get("obstacle_detection") or {}
    tip_pixel = project(hover, config["T_base_camera"], live["intrinsics"])
    if tip_pixel is None:
        return False
    for key in report.get("key_detections", []):
        if key["confidence"] < 0.25:
            continue
        center = (
            np.array(key.get("observed_pixel_roi", key["pixel_roi"]))
            .reshape(2, 2)
            .mean(axis=0)
        )
        elevated = False
        if key.get("position_m") is not None:
            point = np.array(key["position_m"])
            elevated = point[2] - float(table_height_at(point[:2], config)) > 0.025
        moved = np.linalg.norm(center - reference_pixel) > 12
        near_gripper = np.linalg.norm(center - tip_pixel) < 85
        expected_shift = np.linalg.norm(np.array(tip_pixel) - reference_pixel)
        rgb_lift = expected_shift > 20 and np.linalg.norm(
            center - tip_pixel
        ) + 10 < np.linalg.norm(center - reference_pixel)
        if (elevated or rgb_lift) and moved and near_gripper:
            return True
    return False


def run_sequence(
    move,
    gripper,
    verify,
    source,
    hover,
    contact_z,
    stopped,
    emit=lambda _: None,
    max_retries=5,
    normal=None,
    contact_point=None,
):
    """Pure orchestrator: injection makes ordering/retries/Stop testable without a bus."""
    offsets = np.array(
        [[0, 0], [0.002, 0], [-0.002, 0], [0, 0.002], [0, -0.002], [0.001, 0.001]]
    )
    normal = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
    contact_point = (
        np.r_[np.asarray(source)[:2], contact_z]
        if contact_point is None
        else np.asarray(contact_point, float)
    )
    approach_height = max(0.02, float((np.asarray(hover) - contact_point) @ normal))
    for attempt, offset in enumerate(offsets[: max_retries + 1], 1):
        if stopped():
            raise RuntimeError("Grab stopped; arm and gripper hold retained")
        emit(f"Grab {attempt}/{max_retries + 1}: opening")
        gripper(70.0)
        tangent = np.array(
            [
                offset[0],
                offset[1],
                -(normal[0] * offset[0] + normal[1] * offset[1]) / normal[2],
            ]
        )
        down = contact_point + tangent
        above = down + normal * approach_height
        move(above, False)
        emit(f"Grab {attempt}/{max_retries + 1}: descend")
        move(down, True)
        gripper(0.0)
        emit(f"Grab {attempt}/{max_retries + 1}: lift and verify")
        move(above, True)
        if stopped():
            raise RuntimeError("Grab stopped; hold retained")
        if verify(above):
            move(np.array(hover), False)
            return dict(grasp_confirmed=True, attempts=attempt, retries=attempt - 1)
        emit(
            "Keys not confirmed in gripper; retrying"
            if attempt <= max_retries
            else "Keys not confirmed after retry limit; holding"
        )
    return dict(grasp_confirmed=False, attempts=max_retries + 1, retries=max_retries)


def execute_grasp(bus, selected, config, stopped, publish, read_live, sleep=time.sleep):
    from .execute_reach import execute_target, model_pose
    from .grasp_geometry import grasp_settings, oriented_ik, tilt_degrees, tool_geometry
    from .trajectory.execution import lift_route, route_info
    from .trajectory.joint_planner import plan_joint_route

    request = config["grasp_request"]
    source = np.array(request["source_m"])
    hover = np.array(selected.target_m)
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    normal, contact, above, aligned_config, contact_config = grasp_settings(
        arm, config, source, hover
    )
    contact_z = float(contact[2])
    from .trajectory.perception import DetectedBox

    original_info = route_info(selected)
    box_data = (
        original_info["obstacles_m"]
        if original_info
        else read_live().get("obstacle_boxes", [])
    )
    boxes = [
        DetectedBox(
            np.array(b["low_m"]),
            np.array(b["high_m"]),
            b["pixels"],
            b["height_above_table_m"],
            tuple(b["pixel_roi"]),
        )
        for b in box_data
    ]
    revision = (read_live().get("obstacle_detection") or {}).get("scene_revision", 0)
    # Preflight nominal contact/lift before any gripper torque-enable or motion.
    nominal = selected.joints_rad[-1]
    aligned_pose = oriented_ik(
        arm,
        above,
        nominal,
        -normal,
        config["ik_tolerance_m"],
        aligned_config["locked_joints"],
        thorough=True,
    )
    alignment = plan_joint_route(
        arm, nominal, above, aligned_config, boxes, 1, revision, goal_pose=aligned_pose
    )
    preflight = lift_route(
        arm,
        alignment.joints_rad[-1],
        contact,
        np.array([arm.fk(alignment.joints_rad[-1])[:2], contact[:2]]),
        contact_config,
        method="grasp descent",
        candidates=1,
        boxes=boxes,
        scene_revision=revision,
    )
    lift_route(
        arm,
        preflight.joints_rad[-1],
        above,
        np.array([arm.fk(preflight.joints_rad[-1])[:2], above[:2]]),
        contact_config,
        method="grasp lift",
        candidates=1,
        boxes=boxes,
        scene_revision=revision,
    )
    expected_tilt = tilt_degrees(arm, preflight.joints_rad[-1], normal)
    last_plan, last_result = selected, {}

    def emit(message):
        q = model_pose(
            bus.sync_read("Present_Position", list(JOINTS), num_retry=2), config
        )
        goal = model_pose(
            bus.sync_read("Goal_Position", list(JOINTS), num_retry=2), config
        )
        publish(
            dict(
                unix_time_s=time.time(),
                message=message,
                motor_state=dict(
                    q_model_rad=q.tolist(),
                    q_goal_model_rad=goal.tolist(),
                    torque_enabled=bus.sync_read(
                        "Torque_Enable", normalize=False, num_retry=2
                    ),
                ),
            )
        )

    def gripper(percent):
        if percent == 0:
            observed = model_pose(
                bus.sync_read("Present_Position", list(JOINTS), num_retry=2), config
            )
            tip, wrist, axis = tool_geometry(arm, observed)
            if axis @ (-normal) <= 0:
                raise ValueError("Gripper is not pointing down; closing cancelled")
            measured_tilt = tilt_degrees(arm, observed, normal)
            if measured_tilt > expected_tilt + 10:
                raise ValueError(
                    f"Gripper tilt {measured_tilt:.1f}° differs from planned {expected_tilt:.1f}°; closing cancelled"
                )
            emit(
                f"Close: tool tilt {measured_tilt:.1f}°; wrist {(wrist - tip) @ normal * 1000:.1f} mm above tip"
            )
        from .gripper_control import ramp_gripper

        ramp_gripper(bus, percent, stopped, sleep)

    approach_done = False

    def move(target, contact):
        nonlocal approach_done, last_plan, last_result
        if stopped():
            raise RuntimeError("Grab stopped")
        cfg = contact_config if contact else aligned_config
        if not approach_done:
            last_plan, last_result = execute_target(
                bus,
                hover,
                config,
                stopped,
                publish,
                selected_plan=selected,
                sleep=sleep,
                read_scene=read_live,
            )
            approach_done = True
        q = model_pose(
            bus.sync_read("Present_Position", list(JOINTS), num_retry=2), config
        )
        if contact:
            plan = lift_route(
                arm,
                q,
                target,
                np.array([arm.fk(q)[:2], target[:2]]),
                cfg,
                method="grasp vertical",
                candidates=1,
                boxes=boxes,
                scene_revision=revision,
            )
        else:
            goal = oriented_ik(
                arm,
                target,
                q,
                -normal,
                config["ik_tolerance_m"],
                aligned_config["locked_joints"],
                thorough=True,
            )
            plan = plan_joint_route(
                arm, q, target, cfg, boxes, 1, revision, goal_pose=goal
            )
        last_plan, last_result = execute_target(
            bus,
            target,
            cfg,
            stopped,
            publish,
            selected_plan=plan,
            sleep=sleep,
            read_scene=read_live,
        )

    intrinsics = read_live()["intrinsics"]
    reference_pixel = request.get("reference_pixel") or project(
        source, config["T_base_camera"], intrinsics
    )

    def verify(at_hover):
        deadline = time.monotonic() + 8
        count = 0
        seen = set()
        while time.monotonic() < deadline:
            if stopped():
                raise RuntimeError("Grab stopped")
            live = read_live()
            report = live.get("obstacle_detection") or {}
            stamp = report.get("detection_unix_s", 0)
            if stamp not in seen and time.time() - stamp < 3:
                seen.add(stamp)
                count = (
                    count + 1
                    if vision_grasped(live, source, at_hover, config, reference_pixel)
                    else 0
                )
                if count >= 2:
                    return True
            sleep(0.15)
        return False

    result = run_sequence(
        move,
        gripper,
        verify,
        source,
        hover,
        contact_z,
        stopped,
        emit,
        max_retries=min(5, max(0, int(request.get("max_retries", 5)))),
        normal=normal,
        contact_point=contact,
    )
    last_result.update(
        result,
        completed=True,
        gripper_unchanged=False,
        grasp_status="vision confirmed" if result["grasp_confirmed"] else "unconfirmed",
        planned_grasp_tilt_deg=expected_tilt,
        grasp_table_normal=normal.tolist(),
        grasp_contact_m=contact.tolist(),
    )
    return last_plan, last_result
