"""Slow position-mode reach from an existing hold. Does not enable or release torque."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from .adaptive_control import HARDWARE_SPEED_LIMIT_DEG_S
from .config import load_config
from .control_plan import compensated_plan
from .diagnostics import running_tracking_limits
from .feedback_loop import run_feedback
from .kinematics import JOINTS, Arm
from .live_snapshot import atomic_bytes
from .storage import load_plan, save_plan


class MotionStopped(RuntimeError):
    pass


class SceneReplanRequested(MotionStopped):
    """Hold the current goal, then replace only the remaining trajectory."""

    pass


def model_pose(values, config):
    degrees = np.array([values[j] for j in JOINTS], float)
    if not np.isfinite(degrees).all():
        raise ValueError("Nonfinite joint feedback")
    return np.deg2rad(
        degrees * np.asarray(config["joint_signs"]) + config["joint_offsets_deg"]
    )


def motor_command(pose, config):
    values = (np.rad2deg(pose) - config["joint_offsets_deg"]) / np.asarray(
        config["joint_signs"]
    )
    return dict(zip(JOINTS, values.tolist()))


def execute_target(
    bus,
    target,
    config,
    stop_requested,
    publish=lambda _: None,
    clock=time.perf_counter,
    sleep=time.sleep,
    selected_plan=None,
    read_scene=None,
):
    """Continue a requested reach across scene replans; never replay its old start."""
    from .trajectory.execution import RoutePlanner, route_info
    from .trajectory.perception import DetectedBox

    planner = None
    last_report = {}

    def report(value):
        last_report.update(value)
        publish(value)

    replans = 0
    deadline = None
    while True:
        if stop_requested():
            raise MotionStopped("Operator stopped motion; hold retained")
        try:
            plan, result = _execute_target_once(
                bus,
                target,
                config,
                stop_requested,
                report,
                clock,
                sleep,
                selected_plan=selected_plan,
                read_scene=read_scene,
            )
            result["scene_replans"] = replans
            return plan, result
        except SceneReplanRequested:
            if read_scene is None:
                raise
            deadline = deadline or clock() + 45
            original_route = (
                route_info(selected_plan) if selected_plan is not None else None
            )
            method = (original_route or {}).get(
                "trajectory_method", config.get("trajectory_method", "rrt")
            )
            if method not in ("direct", "mlp", "imf", "fm", "rrt"):
                method = "rrt"
            planner = planner or RoutePlanner()
            while True:
                if stop_requested():
                    raise MotionStopped("Operator stopped replanning; hold retained")
                if clock() > deadline:
                    raise MotionStopped(
                        "No clear replacement path after 45 s; hold retained"
                    )
                current = model_pose(
                    bus.sync_read("Present_Position", list(JOINTS), num_retry=2), config
                )
                publish(
                    dict(
                        unix_time_s=time.time(),
                        message="Replanning remaining reach from current pose · resumes automatically",
                        motor_state=dict(
                            last_report.get("motor_state", {}),
                            q_model_rad=current.tolist(),
                        ),
                    )
                )
                try:
                    live = read_scene()
                except (OSError, ValueError):
                    sleep(0.5)
                    continue
                if (live.get("obstacle_detection") or {}).get("error"):
                    sleep(0.5)
                    continue
                if time.time() - live.get("unix_time_s", 0) > 2 or live.get(
                    "camera_stale"
                ):
                    sleep(0.5)
                    continue
                boxes = [
                    DetectedBox(
                        np.array(b["low_m"]),
                        np.array(b["high_m"]),
                        b.get("pixels", 0),
                        b.get("height_above_table_m", 0),
                        tuple(b.get("pixel_roi", [0, 0, 0, 0])),
                    )
                    for b in live.get("obstacle_boxes", [])
                ]
                arm = Arm(
                    config.get("model_joint_limits_deg"), config.get("tool_offset_m")
                )
                try:
                    replacement = planner.plan(
                        arm,
                        current,
                        np.asarray(target),
                        config,
                        boxes,
                        method,
                        config.get("trajectory_candidates", 1),
                        (live.get("obstacle_detection") or {}).get("scene_revision", 0),
                    )
                except ValueError:
                    sleep(0.5)
                    continue
                if stop_requested():
                    raise MotionStopped("Operator stopped replanning; hold retained")
                selected_plan = replacement
                replans += 1
                break


def _execute_target_once(
    bus,
    target,
    config,
    stop_requested,
    publish=lambda _: None,
    clock=time.perf_counter,
    sleep=time.sleep,
    selected_plan=None,
    read_scene=None,
):
    config = deepcopy(config)
    config["max_joint_speed_deg_s"] = min(
        HARDWARE_SPEED_LIMIT_DEG_S, config["max_joint_speed_deg_s"]
    )
    if not config["calibration_verified"] or not config["joint_mapping_verified"]:
        raise ValueError("Camera and model calibration must be verified")
    if not bus.is_calibrated:
        raise ValueError("Motor calibration mismatch")
    names = list(JOINTS)
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    modes = bus.sync_read("Operating_Mode", names, normalize=False, num_retry=2)
    if any(torque[j] != 1 or modes[j] != 0 for j in names):
        raise ValueError("All arm joints must already be holding in position mode")
    temperature = bus.sync_read(
        "Present_Temperature", names, normalize=False, num_retry=2
    )
    if max(temperature.values()) >= 50:
        raise ValueError(
            "A motor is too warm for this commissioning trial; support the arm before releasing hold"
        )
    values = bus.sync_read("Present_Position", names, num_retry=2)
    samples = [model_pose(values, config)]
    for _ in range(2):
        sleep(0.1)
        samples.append(
            model_pose(bus.sync_read("Present_Position", names, num_retry=2), config)
        )
    if np.rad2deg(np.ptp(samples, axis=0)).max() > 0.8:
        raise ValueError("Arm is not holding still; wait for stable feedback before Go")
    observed_start = np.median(samples, axis=0)
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    start = model_pose(bus.sync_read("Goal_Position", names, num_retry=2), config)
    initial_tracking = np.rad2deg(np.abs(start - observed_start))
    running_limits = running_tracking_limits(initial_tracking)
    from .trajectory.execution import check_swept_obstacles, route_info

    route = route_info(selected_plan) if selected_plan is not None else None
    if route:
        from .planning import validate_plan

        plan = deepcopy(selected_plan)
        route = route_info(plan)
        if np.max(np.abs(plan.joints_rad[0] - observed_start)) > np.deg2rad(2):
            raise ValueError("Arm moved since preview; select the target again")
        plan.joints_rad[0] = observed_start
        validate_plan(arm, plan, config)
        if route.get("allow_object_collisions") and not config.get(
            "allow_object_collisions"
        ):
            raise ValueError("Collision override changed; preview the target again")
        if not route.get("allow_object_collisions"):
            check_swept_obstacles(arm, plan.joints_rad, route["obstacles_m"])
        load_bias = observed_start - start
        commands = plan.joints_rad - load_bias
    else:
        plan, commands, load_bias = compensated_plan(
            arm, observed_start, start, target, config
        )
    if stop_requested():
        raise MotionStopped("Stop requested before motion")
    previous = start.copy()
    previous_reference = observed_start.copy()
    began = clock()
    controller = {"command": previous, "reference": previous_reference}
    scene_monitor = None
    if route and read_scene is not None:
        from .trajectory.scene_monitor import SceneMonitor

        scene_monitor = SceneMonitor(
            arm,
            plan,
            route,
            read_scene,
            lambda message: feedback(
                controller.get("estimated", observed_start), message
            ),
        )
    planner_notice_sent = False

    def feedback(measured, message):
        nonlocal planner_notice_sent
        from .learned.solver import planner_status

        notice = (
            ""
            if planner_notice_sent
            else planner_status(plan.planner_info, config.get("planner_mode"))
        )
        planner_notice_sent = True
        publish(
            {
                "unix_time_s": time.time(),
                "message": " | ".join(
                    filter(
                        None,
                        (
                            notice,
                            message,
                        ),
                    )
                ),
                "motor_state": {
                    "q_model_rad": measured.tolist(),
                    "q_filtered_model_rad": controller.get(
                        "estimated", measured
                    ).tolist(),
                    "estimation_std_deg": controller.get(
                        "estimation_std_deg", np.zeros(5)
                    ).tolist(),
                    "torque_enabled": torque,
                    "q_goal_model_rad": controller["command"].tolist(),
                    "q_reference_model_rad": controller["reference"].tolist(),
                    "tracking_limits_deg": running_limits.tolist(),
                    "model_mapping_verified": True,
                },
            }
        )

    def guard_step(current, predicted):
        from .trajectory.joint_planner import check_self_collision

        if scene_monitor is not None:
            from .trajectory.scene_monitor import ScenePathBlocked

            try:
                scene_monitor.poll(current, controller.get("phase_s", 0.0))
            except ScenePathBlocked as exc:
                raise SceneReplanRequested(str(exc)) from exc
        if route:
            check_self_collision(arm, predicted)
        if route:
            from .planning import check_workspace

            check_workspace(arm.fk(predicted), config, label="Corrected tool path")
            if not route.get("allow_object_collisions"):
                try:
                    check_swept_obstacles(
                        arm, [current, predicted], route["obstacles_m"]
                    )
                except ValueError as exc:
                    if scene_monitor is not None:
                        raise SceneReplanRequested(
                            f"Updated object blocks the next step: {exc}"
                        ) from exc
                    raise
        for pose in (current, predicted):
            if not np.isfinite(pose).all():
                raise ValueError("Nonfinite corrected joint state")

        def violation(pose):
            return np.maximum(np.maximum(arm.lower - pose, pose - arm.upper), 0.0)

        if (violation(predicted) - violation(current) > 1e-6).any():
            raise RuntimeError("Corrected step would worsen a joint-limit violation")

        def clearance(pose):
            from .table_collision import minimum_clearance

            return minimum_clearance(arm, pose, config)

        before, after = clearance(current), clearance(predicted)
        if (
            not config.get("allow_object_collisions")
            and after < min(0.0, before) - 0.0001
        ):
            raise RuntimeError(
                "Corrected step would reduce table clearance; position hold retained"
            )

    try:
        observed, trace, max_tracking, packet_stats = run_feedback(
            plan,
            load_bias,
            running_limits,
            lambda: model_pose(
                bus.sync_read("Present_Position", names, num_retry=0), config
            ),
            lambda command: bus.sync_write(
                "Goal_Position", motor_command(command, config), num_retry=0
            ),
            lambda: model_pose(
                bus.sync_read("Goal_Position", names, num_retry=0), config
            ),
            stop_requested,
            controller,
            feedback,
            config["control_hz"],
            clock,
            sleep,
            max_speed_deg_s=config["max_joint_speed_deg_s"],
            joint_lower=arm.lower,
            joint_upper=arm.upper,
            locked=[j in config.get("locked_joints", []) for j in JOINTS],
            guard_step=guard_step,
        )
        end = arm.fk(observed)
        model_gap = float(np.linalg.norm(end - target))
        timing = {}
        for key in ("read_ms", "write_ms", "service_ms", "start_lateness_ms"):
            samples = np.array([row[key] for row in trace])
            timing[key] = {
                "median": float(np.median(samples)),
                "p99": float(np.percentile(samples, 99)),
                "max": float(samples.max()),
            }
        timing["planned_period_ms"] = 1000 / config["control_hz"]
        timing.update(packet_stats)
        result = {
            "completed": True,
            "planner_info": plan.planner_info,
            "target_m": np.asarray(target).tolist(),
            "execution_completed": True,
            "model_target_reached": bool(model_gap <= config["goal_tolerance_m"]),
            "goal_tolerance_mm": config["goal_tolerance_m"] * 1000,
            "settled_model_tool_m": end.tolist(),
            "settled_model_endpoint_error_mm": model_gap * 1000,
            "max_tracking_error_deg": np.rad2deg(max_tracking).tolist(),
            "duration_s": clock() - began,
            "max_command_speed_deg_s": config["max_joint_speed_deg_s"],
            "substeps_per_original_step": 2,
            "control_timing": timing,
            "max_planned_command_speed_deg_s": float(
                np.rad2deg(
                    np.abs(np.diff(commands, axis=0) / np.diff(plan.time_s)[:, None])
                ).max()
            ),
            "initial_tracking_error_deg": initial_tracking.tolist(),
            "initial_load_bias_deg": np.rad2deg(load_bias).tolist(),
            "load_correction_mode": "adaptive_pi",
            "baseline_reestimated_during_move": False,
            "running_tracking_limits_deg": running_limits.tolist(),
            "started_from_existing_hold_target": True,
            "hold_retained": True,
            "gripper_unchanged": True,
            "trace": trace,
            "note": "Tool position is estimated from encoder feedback and the calibrated model; external endpoint accuracy is not measured.",
        }
        feedback(
            observed,
            f"Motion finished. Estimated target gap {model_gap * 1000:.1f} mm "
            f"(tolerance {config['goal_tolerance_m'] * 1000:.0f} mm). Holding; gripper command unchanged.",
        )
        return plan, result
    except BaseException:
        # Retain the last commanded position. Replacing it with load-deflected
        # feedback on every stop can ratchet a gravity-loaded arm downward.
        try:
            values = bus.sync_read("Present_Position", names, num_retry=1)
            observed = model_pose(values, config)
            feedback(
                observed,
                "Reach stopped. Position hold retained; no torque release was commanded.",
            )
        except (OSError, RuntimeError, ValueError):
            pass
        raise
    finally:
        if scene_monitor is not None:
            scene_monitor.close()


def main():
    from .hardware_interlock import require_hardware_enabled

    require_hardware_enabled()
    if Path("runs/control/support_clearance_required").exists():
        raise RuntimeError(
            "Confirm arm holding and hands/support clear in the UI before Go"
        )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--plan", default="runs/last_plan.json")
    p.add_argument("--output", default="runs/control")
    args = p.parse_args()
    config = load_config(args.config)
    selected, metadata = load_plan(args.plan)
    if metadata.get("synthetic"):
        p.error("Cannot execute a synthetic preview")
    for key in ("T_base_camera", "joint_offsets_deg", "joint_signs", "tool_offset_m"):
        if not np.allclose(metadata["config"][key], config[key]):
            p.error(f"Selected preview used different {key}; select a new target")
    if metadata["config"].get("test_bed") != config.get("test_bed"):
        p.error("Selected preview used a different test bed; select a new target")
    control = Path("runs/control")
    control.mkdir(parents=True, exist_ok=True)
    pause = control / "pause_robot"
    stop = control / "stop"
    if stop.exists():
        p.error(
            "A Stop request is active; it must be acknowledged before another reach"
        )
    try:
        with pause.open("x", encoding="utf-8") as stream:
            stream.write(str(time.time()))
    except FileExistsError:
        p.error("Another operation already owns the motor port")
    bus = None

    def publish_report(report):
        try:
            atomic_bytes(control / "state.json", (json.dumps(report) + "\n").encode())
        except OSError:
            pass

    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                live = json.loads(Path("runs/live/state.json").read_text())
                if (
                    live.get("robot_io_paused")
                    and time.time() - live["unix_time_s"] < 2
                ):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(
                "Viewer did not release the motor port; no motion started"
            )
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        robot = SO101Follower(
            SO101FollowerConfig(
                port=config["robot_port"],
                id=config["robot_id"],
                calibration_dir=Path(config["motor_calibration_dir"]),
                use_degrees=True,
            )
        )
        bus = robot.bus
        bus.connect()
        from .hardware_interlock import motion_disabled

        def stopped():
            return stop.exists() or motion_disabled()

        def read_scene():
            return json.loads(Path("runs/live/state.json").read_text(encoding="utf-8"))

        try:
            if config.get("home_request"):
                from .homing import execute_home

                plan, result = execute_home(
                    bus,
                    selected,
                    config,
                    stopped,
                    publish_report,
                    read_scene=read_scene,
                )
            elif config.get("can_request"):
                from .can_discard import execute_can

                plan, result = execute_can(
                    bus, selected, config, stopped, publish_report, read_scene
                )
            elif config.get("grasp_request"):
                from .grasp_keys import execute_grasp

                plan, result = execute_grasp(
                    bus,
                    selected,
                    config,
                    stopped,
                    publish_report,
                    read_scene,
                )
            else:
                plan, result = execute_target(
                    bus,
                    selected.target_m,
                    config,
                    stopped,
                    publish_report,
                    selected_plan=selected,
                    read_scene=read_scene,
                )
        except MotionStopped as exc:
            plan = selected
            result = dict(
                completed=False,
                stopped=True,
                hold_retained=True,
                reason=str(exc),
                reference_only_after_stop=True,
            )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        save_plan(output / "executed_plan.json", plan, config, synthetic=False)
        atomic_bytes(
            output / "result.json", (json.dumps(result, indent=2) + "\n").encode()
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2),
            flush=True,
        )
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        pause.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
