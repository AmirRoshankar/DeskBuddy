import copy
import json
import time
import tkinter as tk
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .adaptive_control import HARDWARE_SPEED_LIMIT_DEG_S
from .calibration_flow import CalibrationFlow
from .calibration_session import (
    assess_captures,
    invalidate_camera,
    remember_session,
    restore_session,
)
from .capture_quality import calibration_depth
from .control_plan import arm_holding, preview_from_feedback
from .demo import synthetic_frame, synthetic_point
from .diagnostics import TRACKING_LIMIT_DEG, tracking_fault
from .geometry import transform_point
from .go_control import launch, read_process_log
from .hardware_interlock import motion_disabled
from .kinematics import Arm
from .learned.solver import planner_status
from .live_snapshot import publish
from .overlays import draw_overlays
from .planning import hover_bounds, plan_reach
from .scene_map import SceneMap
from .storage import save_plan
from .ui.layout import build as build_dashboard
from .ui.theme import COLORS
from .ui.viewport import CameraViewport
from .views import depth_images


def run(config, live=False, robot_state=False, config_path=None):
    """Clicks preview a target; the separate Go action may launch a checked reach."""
    # Endpoint IK remains numerical. Learned models belong to the separate
    # trajectory comparison, not the direct-reach controller.
    config["planner_mode"] = "numerical"
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    scene_map = SceneMap()
    from .trajectory.yolo_perception import YoloService

    object_detector = YoloService()
    from .trajectory.path_overlays import OverlayPlanner

    overlay_planner = OverlayPlanner()
    overlay_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="path-overlays"
    )
    from .trajectory.execution import RoutePlanner, require_same_scene, route_info

    route_planner = RoutePlanner()
    route_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="route")
    route_future = None
    route_version = 0
    config.setdefault("trajectory_method", "rrt")
    retired_mf = config["trajectory_method"] == "mf"
    if retired_mf:
        config["trajectory_method"] = "imf"
    config.setdefault("trajectory_candidates", 1)
    config["allow_object_collisions"] = False
    config["click_and_grab"] = False
    config.pop("grasp_request", None)
    config["home_request"] = False
    scene_window = None
    q = np.deg2rad(config["preview_start_deg"])
    camera = None
    passive = None
    webcam = None
    if live and config.get("secondary_camera"):
        from .webcam import Webcam

        webcam = Webcam(config["secondary_camera"])
    if live:
        from .camera import RealSense
        from .latest_camera import LatestFrameCamera

        camera = LatestFrameCamera(RealSense(config["camera_serial"]))
    if robot_state:
        try:
            from .passive_arm import PassiveArm

            passive = PassiveArm(config)
        except Exception:
            if camera:
                camera.close()
            raise
    root = tk.Tk()
    calibration_mode = passive and not config["calibration_verified"]
    initial_status = (
        (
            "Live camera and passive joint readout. Close the jaws and collect 8 tool poses; no motion commands."
            if calibration_mode
            else ""
        )
        if passive
        else "Offline preview. No robot connection."
    )
    if retired_mf:
        initial_status = "MF retired · iMF selected. " + initial_status

    def request_stop():
        if state.get("auto_calibration"):
            state["auto_calibration"].stop()
        state.pop("auto_grab_token", None)
        path = Path("runs/control/stop")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        status.set("Stop requested. The reach will stop and retain position hold.")

    dashboard = build_dashboard(
        root,
        initial_status=initial_status,
        controls=bool(passive),
        calibration_mode=calibration_mode,
        on_go=lambda: go(),
        on_stop=request_stop,
        on_recalibrate=lambda: recalibrate(),
        on_auto_calibrate=lambda: start_auto_calibration(),
        on_select_tip=lambda: select_tip_feature(),
        hover_m=config["hover_m"],
        on_hover=lambda value: change_hover(value),
        planner_mode=config.get("planner_mode", "numerical"),
        on_planner=lambda value: change_planner(value),
        on_disable=lambda: toggle_disabled(),
        on_bounds=lambda: edit_bounds(),
        on_compare=lambda: open_comparison(),
        trajectory_method=config["trajectory_method"],
        candidates=config["trajectory_candidates"],
        on_trajectory=lambda method, count: change_trajectory(method, count),
        on_scene=lambda: open_scene(),
        on_collision_override=lambda value: change_collision_override(value),
        on_grab_mode=lambda value: change_grab_mode(value),
        on_grab_keys=lambda: (
            discard_can() if config.get("task_mode") == "can_discard" else grab_keys()
        ),
        on_discard_zone=lambda: begin_discard_zone(),
        task_mode=config.get("task_mode", "reach"),
        on_imf_grasp=lambda: imf_grasp(),
        on_hole_grasp=lambda: discard_can(use_hole=True),
        on_record=lambda: toggle_recording(),
        on_home=lambda: home(),
        on_set_home=lambda: set_home(),
        on_calibrate_views=lambda: calibrate_cameras(),
    )
    left, right, depth_view = dashboard.color, dashboard.diagram, dashboard.depth
    webcam_view = dashboard.webcam
    from .ui.recorder import UiRecorder

    recorder = UiRecorder(root)

    def recording_tick():
        if not recorder.recording:
            return
        seconds = int(recorder.elapsed_s())
        dashboard.record.configure(text=f"■ Stop  {seconds // 60}:{seconds % 60:02d}")
        root.after(500, recording_tick)

    def toggle_recording():
        if recorder.recording:
            path = recorder.stop()
            dashboard.record.configure(text="● Record")
            status.set(
                f"Recording saved: {path}"
                if recorder.error is None
                else f"Recording stopped with an error: {recorder.error}"
            )
            return
        path = recorder.start()
        status.set(f"Recording the whole window to {path} · press Stop to save")
        recording_tick()

    zoom, gray_zoom = dashboard.zoom, dashboard.depth_zoom
    status, fault_status, cursor_info = (
        dashboard.status,
        dashboard.fault_status,
        dashboard.cursor_info,
    )
    torque_status, telemetry_status = (
        dashboard.torque_status,
        dashboard.telemetry_status,
    )
    go_button, help_label = dashboard.go, dashboard.help_label
    display_w, display_h, _diagram_h = (
        dashboard.display_w,
        dashboard.display_h,
        dashboard.diagram_h,
    )
    state = {
        "q": q,
        "frame": None,
        "plan": None,
        "anim": None,
        "last_publish": 0.0,
        "motor": None,
        "last_state_read": 0.0,
        "history": [],
        "tool_points": [],
        "webcam_points": [],
        "session_id": str(uuid.uuid4()),
        "depth_history": [],
        "cursor": None,
        "selected_pixel": None,
        "selected_depth": None,
        "motion_job": None,
        "close_requested": False,
        "last_telemetry_logged": 0.0,
        "refresh_job": None,
        "surface_point_m": None,
        "selection_executed": False,
        "visual_check": None,
        "calibration_job": None,
        "secondary_frame": None,
        "path_overlays": [],
        "overlay_future": None,
    }
    state["robot_io_paused"] = False
    if calibration_mode:
        session_id, rows = restore_session(config)
        if session_id:
            state["session_id"] = session_id
            state["tool_points"] = rows
    live_dir = Path("runs/live")
    if camera:
        live_dir.mkdir(parents=True, exist_ok=True)
        (live_dir / "stop").unlink(missing_ok=True)

    def refresh_arm_model():
        nonlocal arm
        arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))

    def select_camera(event, secondary=False):
        if event.state & 0x4:
            return "break"
        if (
            state["motion_job"] is not None
            or state["robot_io_paused"]
            or state.get("calibration_job") is not None
        ):
            return "break"
        try:
            pixel = event.widget.viewport.to_camera(event.x, event.y)
            if pixel is None:
                return "break"
            if secondary:
                from .multiview import calibration_current, secondary_surface_point

                if calibration_mode:
                    if calibration_flow.stage == "capture":
                        capture_webcam_tip(pixel)
                        return "break"
                    if calibration_flow.stage == "paper_webcam":
                        calibration_flow.capture_paper_webcam(pixel)
                        return "break"
                    raise ValueError(
                        "follow the current calibration step in the bar below"
                    )
                if not calibration_current(config):
                    raise ValueError("Calibrate views before selecting from the webcam")
                point = secondary_surface_point(
                    pixel, config["secondary_camera_calibration"], bounded=False
                )
                primary_pixel = None
            else:
                from .test_bed import surface_point

                if state["frame"] is None:
                    raise ValueError("Wait for a camera frame")
                pick_config = (
                    {**config, "restrict_to_test_bed": False}
                    if state.get("discard_points") is not None
                    else config
                )
                point = surface_point(*pixel, state["frame"][2], pick_config)
                primary_pixel = pixel
            state["selection_in_3d"] = True
            accept_world_selection(point, primary_pixel)
        except (ValueError, RuntimeError) as exc:
            status.set(f"{'Webcam' if secondary else 'Camera'} target rejected: {exc}")
        return "break"

    def capture_webcam_tip(pixel):
        """RGB-only arm point: jaw-tip click in the webcam + encoder FK (no depth)."""
        from .rgb_hand_eye import fit_camera

        frame = state.get("secondary_frame")
        if frame is None or state.get("camera_stale"):
            raise ValueError("Wait for a fresh webcam frame")
        motor = state["motor"]
        if motor is None or time.monotonic() - state["last_state_read"] > 0.5:
            raise ValueError("Wait for joint readings")
        if any(motor["torque_enabled"].values()):
            raise ValueError(
                "Manual capture requires every motor's torque to be disabled"
            )
        if motor["gripper_percent"] > 8:
            raise ValueError(
                "Gently close the gripper jaws by hand before clicking their meeting point"
            )
        recent = [
            item[1] for item in state["history"] if time.monotonic() - item[0] < 0.7
        ]
        if len(recent) < 3 or np.rad2deg(np.ptp(recent, axis=0)).max() > 0.7:
            raise ValueError("Hold the arm still for one second, then click again")
        q_capture = np.array(motor["q_model_rad"])
        if (q_capture < arm.lower).any() or (q_capture > arm.upper).any():
            raise ValueError(
                "Move the joints away from their end stops into a comfortable middle pose"
            )
        base = arm.fk(q_capture)
        if any(
            np.linalg.norm(base - r["base_point_model_m"]) < 0.025
            for r in state["webcam_points"]
        ):
            raise ValueError(
                "Choose a new tool position at least 2.5 cm from earlier samples"
            )
        image = frame[0]
        count = len(state["webcam_points"]) + 1
        capture_dir = Path("runs/calibration") / state["session_id"]
        capture_dir.mkdir(parents=True, exist_ok=True)
        marked = Image.fromarray(image)
        mark = ImageDraw.Draw(marked)
        x, y = pixel
        mark.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#ff4060", width=2)
        mark.text((10, 10), f"Webcam point {count}", fill="#ff4060")
        rgb_path = capture_dir / f"webcam_point_{count:02d}.png"
        marked.save(rgb_path)
        record = dict(
            session_id=state["session_id"],
            unix_time_s=time.time(),
            view="webcam",
            pixel=[float(x), float(y)],
            image_size=[image.shape[1], image.shape[0]],
            base_point_model_m=base.tolist(),
            q_model_rad=q_capture.tolist(),
            gripper_percent=motor["gripper_percent"],
            rgb_file=str(rgb_path),
        )
        with Path("runs/webcam_tool_points.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record) + "\n")
        state["webcam_points"].append(record)
        count = len(state["webcam_points"])
        if count < 12:
            calibration_flow.capture_step()
            status.set(
                f"Webcam arm point {count}/12 captured. Move to a new position and height, "
                "hold still and click the jaw tip in the webcam again."
            )
            return
        try:
            fit = fit_camera(state["webcam_points"][:12], record["image_size"])
        except ValueError as exc:
            calibration_flow.bar.step(
                f"Webcam fit needs new points: {exc}", "Restart arm points", recalibrate
            )
            return
        from .calibration_session import write_json

        fit["image_size"] = record["image_size"]
        write_json(capture_dir / "webcam_fit.json", fit)
        calibration_flow.webcam_assessment(fit)

    webcam_view.bind("<Button-1>", lambda event: select_camera(event, True))

    def calibrate_cameras():
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Finish the motor operation before aligning camera views.")
            return
        if state["frame"] is None or state["secondary_frame"] is None:
            status.set("Wait for fresh images from both cameras.")
            return
        existing = state.get("camera_alignment_window")
        if existing is not None and not existing.closed:
            existing.window.deiconify()
            existing.window.lift()
            existing.window.focus_force()
            return
        try:
            from .ui.camera_alignment import CameraAlignment
        except Exception as exc:
            status.set(f"Cannot open camera alignment: {type(exc).__name__}: {exc}")
            return

        rgb, depth, intrinsics = state["frame"]

        def save_alignment(result):
            if result.get("camera_only"):
                config["camera_pair_alignment"] = result
                if config_path:
                    from .calibration_session import write_json

                    write_json(config_path, config)
                status.set(
                    "Camera alignment saved on A4. Robot-base calibration is still needed for motion."
                )
                return
            refinement = result.get("primary_camera_refinement")
            if refinement:
                # Keep the previous pose so the change can be undone by hand.
                backup = Path("runs/calibration-backups")
                backup.mkdir(parents=True, exist_ok=True)
                (
                    backup / f"T_base_camera-{time.strftime('%Y%m%d-%H%M%S')}.json"
                ).write_text(
                    json.dumps(dict(T_base_camera=config["T_base_camera"]), indent=2),
                    encoding="utf-8",
                )
                config["T_base_camera"] = refinement["T_base_camera"]
            config["test_bed"] = result["primary_bed"]
            config["secondary_camera_calibration"] = result
            if config_path:
                from .calibration_session import write_json

                write_json(config_path, config)
            change_hover(config["hover_m"])
            status.set(
                "Alignment saved · "
                + (
                    f"RealSense tilt corrected {refinement['rotation_change_deg']:.1f}° "
                    "from the A4 · "
                    if refinement
                    else ""
                )
                + result.get("pose_validation", "table-plane support")
            )

        def frames_provider():
            if (
                state["frame"] is None
                or state["secondary_frame"] is None
                or state.get("camera_stale")
            ):
                return None
            return (
                state["frame"][0].copy(),
                state["secondary_frame"][0].copy(),
                None if state["frame"][1] is None else state["frame"][1].copy(),
            )

        try:
            state["camera_alignment_window"] = CameraAlignment(
                root,
                rgb.copy(),
                state["secondary_frame"][0].copy(),
                None if depth is None else depth.copy(),
                intrinsics,
                copy.deepcopy(config),
                save_alignment,
                frames_provider=frames_provider,
            )
            status.set(
                "Camera alignment opened. Select matching corners in both views."
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            status.set(f"Cannot open camera alignment: {type(exc).__name__}: {exc}")

    def current_hole():
        """Pour-hole centre from the active plan, else the latest detection."""
        plan = state.get("plan")
        for info in (plan.planner_info or []) if plan is not None else []:
            proposal = (info.get("can_transfer") or {}).get("model_proposal") or {}
            if proposal.get("hole_center_m"):
                return proposal["hole_center_m"]
        opening = state.get("can_opening")
        return None if opening is None else opening["center_m"]

    def scene_snapshot():
        from .can_discard import HOLE_TIP_DEPTH_M

        plan = state.get("plan")
        objective = None
        transfer = next(
            (
                i["can_transfer"]
                for i in (plan.planner_info or [] if plan is not None else [])
                if "can_transfer" in i
            ),
            None,
        )
        proposal = (transfer or {}).get("model_proposal") or {}
        if proposal.get("hole_center_m"):
            objective = dict(
                point_m=proposal["hole_center_m"],
                depth_m=proposal["hole_center_m"][2]
                - (
                    transfer["can"]["center_m"][2]
                    + transfer["can"]["height_m"]
                    - HOLE_TIP_DEPTH_M
                ),
                label="Pour hole · objective",
            )
        elif state.get("can_opening") is not None:
            objective = dict(
                point_m=state["can_opening"]["center_m"],
                depth_m=0.0,
                label="Pour hole (detected)",
            )
        return dict(
            objective=objective,
            target_label="Drop" if transfer else "Target",
            arm=arm,
            q=np.array(state["q"]).copy(),
            config=config,
            boxes=state.get("obstacle_boxes", []),
            plan=plan,
            paths=state.get("path_overlays", []),
            selected_method=config["trajectory_method"],
            target=plan.target_m if plan is not None else None,
            discard_points=state.get("discard_points"),
            pose_source="encoder estimate"
            if state.get("motor")
            else "last/preview pose · live encoders unavailable",
        )

    def open_scene():
        nonlocal scene_window
        if scene_window is not None and scene_window.window.winfo_exists():
            scene_window.window.lift()
            return
        from .ui.scene_inspector import SceneInspector

        scene_window = SceneInspector(root, scene_snapshot)

    def change_collision_override(value):
        config["allow_object_collisions"] = bool(value)
        if state["motion_job"] is not None or state["robot_io_paused"]:
            if not value:
                request_stop()
            status.set("Collision setting applies to the next preview.")
            return
        change_hover(config["hover_m"])

    def set_home():
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set(
                "Cannot save Home while another operation owns the motor port; wait for feedback to resume."
            )
            return
        motor = state.get("motor")
        if motor is None or time.monotonic() - state["last_state_read"] > 0.5:
            status.set("Cannot save Home: wait for fresh joint readings.")
            return
        pose = np.asarray(motor.get("q_model_rad", []), float)
        if pose.shape != (5,) or not np.isfinite(pose).all():
            status.set("Cannot save Home: invalid joint readings.")
            return
        if (pose < arm.lower).any() or (pose > arm.upper).any():
            status.set(arm.limit_error(pose, "Home pose"))
            return
        candidate = copy.deepcopy(config)
        candidate["home_pose_deg"] = np.rad2deg(pose).tolist()
        try:
            if config_path:
                from .calibration_session import write_json

                write_json(config_path, candidate)
            config.update(candidate)
            status.set(
                "Home saved [pan, lift, elbow, wrist, roll]: "
                + ", ".join(f"{angle:.1f} deg" for angle in candidate["home_pose_deg"])
                + (" (saved to disk)." if config_path else " (session only).")
                + " No movement commanded."
            )
        except OSError as exc:
            status.set(f"Home was not saved: {exc}")

    def home():
        nonlocal route_version, route_future
        if not passive or state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Wait for the current operation before Home.")
            return
        if state.get("comparison_open"):
            status.set("Close Compare before Home.")
            return
        if not arm_holding(state.get("motor")) or motion_disabled():
            status.set("Press Enable before Home.")
            return
        from .homing import plan_home

        route_version += 1
        if route_future is not None:
            route_future.cancel()
        token = uuid.uuid4().hex
        state["auto_grab_token"] = token
        state["plan"] = None
        if go_button:
            go_button.configure(state="disabled")
        cfg = copy.deepcopy(config)
        cfg["max_joint_speed_deg_s"] = min(
            HARDWARE_SPEED_LIMIT_DEG_S, cfg["max_joint_speed_deg_s"]
        )
        boxes = copy.deepcopy(state.get("obstacle_boxes", []))
        revision = (state.get("obstacle_report") or {}).get("scene_revision", 0)
        future = route_executor.submit(
            plan_home, arm, np.array(state["q"]).copy(), cfg, boxes, revision
        )
        route_future = future
        status.set("Planning Home…")

        def ready():
            if state.get("auto_grab_token") != token:
                return
            if not future.done():
                root.after(75, ready)
                return
            state.pop("auto_grab_token", None)
            try:
                plan, job_config = future.result()
                require_same_scene(
                    route_info(plan),
                    dict(
                        unix_time_s=time.time(),
                        camera_stale=state.get("camera_stale"),
                        obstacle_detection=state.get("obstacle_report"),
                        obstacle_boxes=[
                            b.as_dict() for b in state.get("obstacle_boxes", [])
                        ],
                    ),
                    arm=arm,
                    joints=plan.joints_rad,
                )
                state["plan"] = plan
                state["motion_job"] = launch(plan, job_config, state["motor"])
                state["selection_executed"] = True
                status.set("Home: moving to rest pose, then closing gripper.")
                root.after(100, poll_motion)
            except (ValueError, RuntimeError, OSError) as exc:
                status.set(str(exc))

        root.after(75, ready)

    def begin_discard_zone():
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Stop motion before calibrating the discard zone")
            return
        clear_paths()
        state["discard_points"] = []
        status.set(
            "Discard zone: click four table-plane corners in any view; Ctrl navigates without selecting"
        )

    def accept_world_selection(point, pixel=None, requested_hover=None):
        if state.get("discard_points") is None:
            preview_point(point, pixel, requested_hover=requested_hover)
            return
        points = state["discard_points"]
        points.append(np.asarray(point).tolist())
        if len(points) < 4:
            status.set(f"Discard zone {len(points)}/4 corners")
            return
        try:
            from .can_discard import zone_points

            zone = zone_points(points, config["table_z_m"])
            config["discard_zone"] = zone.tolist()
            # Selecting the drop area explicitly extends only XY; preserve Z.
            config["workspace_min_m"][:2] = np.minimum(
                config["workspace_min_m"][:2], zone[:, :2].min(0) - 0.01
            ).tolist()
            config["workspace_max_m"][:2] = np.maximum(
                config["workspace_max_m"][:2], zone[:, :2].max(0) + 0.01
            ).tolist()
            config["restrict_to_test_bed"] = False
            if config_path:
                from .calibration_session import write_json

                write_json(config_path, config)
            state["discard_points"] = None
            status.set("Discard zone saved · XY bounds include it; Z limits unchanged")
        except ValueError as exc:
            state["discard_points"] = []
            status.set(f"Discard zone: {exc}; select four corners again")

    def discard_can(use_imf=False, use_hole=False):
        nonlocal route_version, route_future
        if not passive or state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Wait for live motor feedback and an idle arm")
            return
        if motion_disabled() or not arm_holding(state.get("motor")):
            status.set("Press Enable before discarding a can")
            return
        if config.get("discard_zone") is None:
            begin_discard_zone()
            return
        from .can_discard import detect_can, learned_can_cycle, plan_can_cycle

        state["can_opening"] = None
        try:
            report = state.get("obstacle_report") or {}
            if time.time() - report.get("detection_unix_s", 0) > 3:
                raise ValueError("Wait for fresh two-view can detection")
            boxes = state.get("obstacle_boxes", [])
            raw = report.get("raw_obstacle_boxes", [])
            if boxes and raw:
                from .trajectory.perception import DetectedBox

                boxes = [
                    DetectedBox(
                        np.array(b["low_m"]),
                        np.array(b["high_m"]),
                        b.get("pixels", 0),
                        b.get("height_above_table_m", 0),
                        tuple(b.get("pixel_roi", [0, 0, 0, 0])),
                        confidence=b.get("confidence"),
                    )
                    for b in raw
                    if any(
                        np.linalg.norm(
                            (
                                np.array(b["low_m"][:2])
                                + np.array(b["high_m"][:2])
                                - old.low[:2]
                                - old.high[:2]
                            )
                            / 2
                        )
                        < 0.05
                        for old in boxes
                    )
                ] or boxes
            can = detect_can(boxes, config)
            opening = None
            if use_hole:
                from .trajectory.can_opening import detect_webcam_opening

                secondary = state.get("secondary_frame")
                if secondary is None:
                    raise ValueError("Need a webcam RGB frame to find the pour hole")
                webcam_rgb = secondary[0] if isinstance(secondary, tuple) else secondary
                opening = detect_webcam_opening(webcam_rgb, config, can)
                state["can_opening"] = opening
                gap = opening.get("lid_views_m", {}).get("disagreement_m")
                status.set(
                    f"Pour hole found ({opening['confidence']:.2f}); "
                    f"lid centre corrected {np.linalg.norm(opening['lid_correction_m']) * 1000:.0f} mm"
                    + (
                        f" · cameras differ {gap * 1000:.0f} mm, using RealSense; "
                        "jaw camera checks will correct"
                        if gap is not None and gap > 0.02
                        else ""
                    )
                )
        except ValueError as exc:
            status.set(str(exc))
            return
        clear_paths()
        route_version += 1
        version = route_version
        token = uuid.uuid4().hex
        state["auto_grab_token"] = token
        cfg = copy.deepcopy(config)
        seed = np.array(state["q"]).copy()
        if use_hole:
            from .can_discard import hole_can_cycle

            route_future = route_executor.submit(
                hole_can_cycle, arm, seed, cfg, can, opening
            )
        else:
            route_future = route_executor.submit(
                learned_can_cycle if use_imf else plan_can_cycle, arm, seed, cfg, can
            )
        status.set(
            "Planning "
            + ("pour-hole " if use_hole else "iMF " if use_imf else "")
            + "can pick/lift/drop · Stop cancels before execution"
        )

        def ready():
            if version != route_version or state.get("auto_grab_token") != token:
                return
            if not route_future.done():
                root.after(100, ready)
                return
            try:
                plan, job_config = route_future.result()
                if motion_disabled():
                    raise ValueError("Hardware disabled")
                state["plan"] = plan
                show(right, diagram(state["q"], plan=plan))
                state["motion_job"] = launch(plan, job_config, state["motor"])
                state.pop("auto_grab_token", None)
                status.set("Discard can: open → grasp → lift → discard zone → release")
                root.after(100, poll_motion)
            except (ValueError, RuntimeError, OSError) as exc:
                clear_paths()
                status.set(f"Can sequence unavailable: {exc}")

        root.after(100, ready)

    def imf_grasp():
        if config.get("task_mode") == "can_discard":
            discard_can(use_imf=True)
            return
        from .ui.grasp_candidates import GraspCandidates

        GraspCandidates(
            root,
            lambda: dict(
                config=copy.deepcopy(config),
                boxes=copy.deepcopy(state.get("obstacle_boxes", [])),
                report=copy.deepcopy(state.get("obstacle_report") or {}),
                selected=state.get("surface_point_m"),
                intrinsics=state["frame"][2]
                if state.get("frame") is not None
                else None,
            ),
        )

    def grab_keys():
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Wait for the current motion to finish.")
            return
        try:
            from .grasp_keys import detect_key_target

            report = state.get("obstacle_report") or {}
            if time.time() - report.get("detection_unix_s", 0) > 3:
                raise ValueError("Wait for fresh key detection")
            point, key = detect_key_target(report, state["frame"][2], config)
            roi = key["pixel_roi"]
            pixel = key.get(
                "grasp_pixel", ((roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2)
            )
            preview_point(point, pixel, for_grasp=True)
            version = route_version
            token = uuid.uuid4().hex
            state["auto_grab_token"] = token
            deadline = time.monotonic() + 30

            def ready():
                if version != route_version or state.get("auto_grab_token") != token:
                    return
                if state["plan"] is None:
                    if time.monotonic() < deadline:
                        root.after(100, ready)
                    return
                state["auto_grasp_source"] = point.tolist()
                state["auto_grasp_reference_pixel"] = [
                    (roi[0] + roi[2]) / 2,
                    (roi[1] + roi[3]) / 2,
                ]
                state.pop("auto_grab_token", None)
                go()
                state.pop("auto_grasp_source", None)
                state.pop("auto_grasp_reference_pixel", None)

            root.after(100, ready)
        except (ValueError, RuntimeError, OSError) as exc:
            status.set(str(exc))

    def toggle_disabled():
        state.pop("auto_grab_token", None)
        if state.get("auto_calibration"):
            state["auto_calibration"].stop()
        if state.get("release_pending"):
            status.set("Torque release in progress · keep arm supported")
            return
        holding = arm_holding(state["motor"])
        busy = (
            state["motion_job"] is not None
            or state["calibration_job"] is not None
            or state["robot_io_paused"]
        )
        if holding or busy:
            clear_paths()
            hold_control.release()
        elif calibration_flow.stage is not None:
            status.set("Finish the calibration step before enabling.")
        else:
            clear_paths()
            hold_control.start(confirmed=True)
        dashboard.disable.configure(text="Enable" if holding or busy else "Disable")

    bounds_window = None

    def edit_bounds():
        nonlocal bounds_window
        from .ui.workspace import open_editor

        if bounds_window is not None and bounds_window.winfo_exists():
            bounds_window.lift()
            return

        def apply_bounds(updated):
            if (
                state.get("bounds_drag") is not None
                or state["motion_job"] is not None
                or state["calibration_job"] is not None
                or state["robot_io_paused"]
            ):
                raise ValueError("Stop the current operation before changing bounds")
            if config_path:
                from .calibration_session import write_json

                write_json(Path(config_path), updated)
            config.update(updated)
            state["plan"] = None
            state["surface_point_m"] = None
            state["selected_pixel"] = None
            if go_button:
                go_button.configure(state="disabled")
            status.set("Playground bounds saved. Select a target to preview.")

        bounds_window = open_editor(root, config, apply_bounds)

    def calibration_finished():
        nonlocal calibration_mode, scene_map
        calibration_mode = False
        scene_map = SceneMap()
        help_label.grid_remove()
        if state.get("hold_pending_clear"):
            hold_control.finish()

    def cancel_recalibration():
        """Restore the camera/table calibration saved when Recalibrate began.

        Calibration data only: torque/hold state is unchanged, so if the arm
        was released for hand-guiding it stays released until Enable.
        """
        from tkinter import messagebox

        if state.get("calibration_job") is not None:
            status.set("Wait for the current motor check to finish.")
            return
        backup = Path("runs/calibration_before_reset.json")
        try:
            previous = json.loads(backup.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status.set("No saved previous calibration to restore.")
            return
        if not previous.get("T_base_camera") or not previous.get("test_bed"):
            status.set(
                "The saved previous calibration is incomplete; finish Recalibrate."
            )
            return
        if not messagebox.askyesno(
            "Restore previous calibration",
            "Discard this recalibration and restore the previous camera and A4 "
            "calibration? Motor torque is not changed: if the arm is released, "
            "support it and press Enable afterwards.",
            parent=root,
        ):
            return
        for key in (
            "T_base_camera",
            "test_bed",
            "tool_offset_m",
            "calibration_verified",
        ):
            if key in previous:
                config[key] = previous[key]
        if config_path:
            from .calibration_session import write_json

            write_json(config_path, config)
        refresh_arm_model()
        calibration_flow.stage = None
        calibration_flow.bar.grid_remove()
        calibration_finished()
        status.set(
            "Previous calibration restored. Torque unchanged"
            + (
                " · arm released: support it and press Enable."
                if state.get("motor")
                and not all(state["motor"]["torque_enabled"].values())
                else "."
            )
        )

    calibration_flow = CalibrationFlow(
        root,
        config,
        state,
        status,
        config_path,
        refresh_arm_model,
        calibration_finished,
        lambda: recalibrate(),
    )
    calibration_flow.bar.on_cancel = cancel_recalibration
    from .hold_control import HoldControl

    hold_control = HoldControl(root, config, state, status, calibration_flow.bar)

    def open_manual_rgb_capture():
        from .calibration_session import write_json
        from .rgb_hand_eye import fit_camera
        from .ui.paired_calibration import PairedCalibration

        def frames():
            if state.get("frame") is None or state.get("secondary_frame") is None:
                return None
            return state["frame"][0].copy(), state["secondary_frame"][0].copy()

        def capture_state():
            motor = state.get("motor")
            if not motor or time.monotonic() - state["last_state_read"] > 0.5:
                raise ValueError("Wait for fresh joint readings")
            if any(motor["torque_enabled"].values()):
                raise ValueError(
                    "Support the arm and release hold using the calibration controls first"
                )
            if motor["gripper_percent"] > 8:
                raise ValueError(
                    "Close the jaws by hand before selecting their meeting point"
                )
            recent = [
                item[1] for item in state["history"] if time.monotonic() - item[0] < 0.7
            ]
            if len(recent) < 3 or np.rad2deg(np.ptp(recent, axis=0)).max() > 0.7:
                raise ValueError("Hold the arm still for one second")
            q_sample = np.array(motor["q_model_rad"])
            base = arm.fk(q_sample)
            if any(
                np.linalg.norm(base - np.asarray(r["base_point_model_m"])) < 0.025
                for r in state["webcam_points"]
            ):
                raise ValueError("Move the tip at least 2.5 cm from earlier poses")
            return dict(
                q_model_rad=q_sample.tolist(),
                base_point_model_m=base.tolist(),
                gripper_percent=motor["gripper_percent"],
            )

        primary_records = []

        def save_pair(pixels, context, images):
            records = [
                dict(
                    context,
                    pixel=list(pixel),
                    image_size=[image.shape[1], image.shape[0]],
                    session_id=state["session_id"],
                    unix_time_s=time.time(),
                    view=view,
                )
                for pixel, image, view in zip(pixels, images, ("realsense", "webcam"))
            ]
            primary_records.append(records[0])
            state["webcam_points"].append(records[1])
            directory = Path("runs/calibration") / state["session_id"]
            write_json(
                directory / "paired_rgb_points.json",
                dict(primary=primary_records, secondary=state["webcam_points"]),
            )
            count = len(primary_records)
            status.set(f"Manual calibration: {count}/8 pairs ({count * 2}/16 clicks)")
            if count == 8:
                try:
                    fits = [
                        fit_camera(rows, record["image_size"])
                        for rows, record in (
                            (primary_records, records[0]),
                            (state["webcam_points"], records[1]),
                        )
                    ]
                    for fit, record in zip(fits, records):
                        fit["image_size"] = record["image_size"]
                    write_json(
                        directory / "paired_rgb_fits.json",
                        dict(primary=fits[0], secondary=fits[1]),
                    )
                    calibration_flow.webcam_assessment(fits[1])
                    status.set(
                        "8 RGB tip pairs captured; camera fit results ready for review."
                    )
                except ValueError as exc:
                    calibration_flow.bar.step(
                        f"Paired camera fit needs new points: {exc}",
                        "Restart arm points",
                        recalibrate,
                    )
                    status.set(f"8 pairs captured; fit failed: {exc}")

        if frames() is None:
            status.set("Connect both cameras before paired manual calibration.")
            return
        state["webcam_points"] = []
        state["manual_rgb_window"] = PairedCalibration(
            root, frames, capture_state, save_pair
        )

    calibration_flow.on_manual_capture = open_manual_rgb_capture

    def select_tip_feature():
        from .point_tracking import PointTracker
        from .ui.tip_picker import TipPicker

        if state.get("motion_job") is not None:
            status.set("Stop motion before selecting the fixed jaw tip.")
            return
        if state.get("secondary_frame") is None:
            status.set("Wait for the webcam image.")
            return

        def selected(image, pixel, stamp):
            state["tip_tracker"] = PointTracker(image, pixel, stamp)
            state["tip_tracking"] = state["tip_tracker"].result()
            status.set(
                "Fixed jaw tip selected. Tracking uses image motion, not orange colour. Small checked probes run at the next can pregrasp."
            )

        TipPicker(root, lambda: state.get("secondary_frame"), selected)

    def start_auto_calibration():
        from .ui.auto_calibration import AutoCalibration

        if (
            state.get("calibration_job") is not None
            or state.get("motion_job") is not None
        ):
            status.set(
                "Wait for the current motor operation before automatic calibration."
            )
            return
        # Switching from manual capture must not restore an older transform or
        # require completing the very calibration the automatic routine replaces.
        calibration_flow.stage = None
        calibration_flow.bar.grid_remove()
        manual_window = state.get("manual_rgb_window")
        if manual_window is not None and not manual_window.closed:
            manual_window.close()
        try:
            operation = AutoCalibration(
                root, config, state, arm, status, launch, route_executor
            )
            state["auto_calibration"] = operation
            operation.start()
        except (ValueError, RuntimeError, OSError) as exc:
            status.set(f"Automatic calibration unavailable: {exc}")

    comparison_window = None

    def open_comparison():
        nonlocal comparison_window
        if not camera or state["frame"] is None:
            status.set("Live comparison needs the calibrated camera and arm readout.")
            return
        if (
            state["motion_job"] is not None
            or state["calibration_job"] is not None
            or state["robot_io_paused"]
        ):
            status.set("Finish the current operation before comparing paths.")
            return
        if comparison_window is not None and comparison_window.window.winfo_exists():
            comparison_window.window.lift()
            return
        from copy import deepcopy

        from .ui.comparison import ComparisonWindow

        state["comparison_open"] = True
        state["plan"] = None
        state["surface_point_m"] = None
        state["selected_pixel"] = None
        if go_button:
            go_button.configure(state="disabled")

        def provider():
            if not config["calibration_verified"]:
                raise ValueError("Calibrate camera and table first")
            rgb, depth, intrinsics = state["frame"]
            provider.arm = arm
            provider.perception_ms = state.get("obstacle_report", {}).get(
                "processing_ms"
            )
            return (
                rgb.copy(),
                depth,
                intrinsics,
                deepcopy(config),
                state["q"].copy(),
                list(state.get("obstacle_boxes", [])),
            )

        provider.arm = arm

        def closed():
            nonlocal comparison_window
            state["comparison_open"] = False
            state["plan"] = None
            comparison_window = None
            if go_button:
                go_button.configure(state="disabled")

        comparison_window = ComparisonWindow(root, provider, closed)

    def assess_if_ready():
        try:
            assessment = assess_captures(state["tool_points"])
            if assessment:
                state["calibration_report"] = assessment[0]
                status.set(assessment[1])
                calibration_flow.assessment(assessment[1])
        except (ValueError, OSError) as exc:
            status.set(
                f"8 points captured — automatic calibration check failed: {exc}. Go remains disabled."
            )
            calibration_flow.bar.step(
                f"Arm fit needs new points: {exc}", "Restart arm points", recalibrate
            )

    def recalibrate():
        nonlocal calibration_mode, scene_map
        if (
            state["motion_job"] is not None
            or state["calibration_job"] is not None
            or state["robot_io_paused"]
            or Path("runs/control/pause_robot").exists()
        ):
            status.set(
                "Stop motion and wait for it to finish before recalibrating. Hold is retained."
            )
            return
        try:
            invalidate_camera(config, config_path)
            calibration_mode = True
            scene_map = SceneMap()
            state.update(
                plan=None,
                selected_pixel=None,
                surface_point_m=None,
                visual_check=None,
                tool_points=[],
                webcam_points=[],
                session_id=str(uuid.uuid4()),
                selection_executed=False,
                calibration_report=None,
            )
            remember_session(state["session_id"], config)
            go_button.configure(state="disabled")
            help_label.grid_remove()
            calibration_flow.begin()
            status.set(
                "Recalibration started: previous camera transform invalidated; Go disabled. Motor hold unchanged. Support the arm before releasing torque for hand-guided capture."
            )
        except (OSError, ValueError) as exc:
            go_button.configure(state="disabled")
            status.set(f"Recalibration could not be saved: {exc}. Go disabled.")

    if calibration_mode and len(state["tool_points"]) >= 8:
        help_label.grid_remove()
        assess_if_ready()
        calibration_flow.stage = "capture"
    elif calibration_mode:
        help_label.grid_remove()
        calibration_flow.begin()

    def show(label, image):
        if label is right and getattr(label, "rendered_source", None) is image:
            return
        if label is right:
            label.rendered_source = image
        if label in (zoom, gray_zoom):
            label.zoom_source = image
            side = getattr(label, "zoom_side", 144)
            image = image.resize((side, side), Image.Resampling.NEAREST)
        if label in (left, depth_view, webcam_view):
            width, height = label.winfo_width(), label.winfo_height()
            if min(width, height) < 32:
                width, height = display_w, display_h
            label.viewport = CameraViewport(width, height, image.width, image.height)
            image = label.viewport.render(image)
        label.image = ImageTk.PhotoImage(image)
        label.configure(image=label.image)

    scene_cache = {}

    def diagram(pose, target=None, plan=None):
        from .ui.scene_inspector import render_scene

        snapshot = scene_snapshot()
        size = (max(160, right.winfo_width()), max(120, right.winfo_height()))
        nav = scene_nav.view
        key = (
            size,
            tuple(snapshot["q"]),
            id(plan),
            repr(snapshot["boxes"]),
            repr(snapshot["paths"]),
            id(nav),
            nav.yaw,
            nav.pitch,
            nav.zoom,
            tuple(nav.pan),
            snapshot["pose_source"],
            repr(config.get("discard_zone")),
            repr(state.get("discard_points")),
            repr(state.get("can_opening")),
            tuple(config["workspace_min_m"]),
            tuple(config["workspace_max_m"]),
        )
        if scene_cache.get("key") != key:
            scene_cache["image"] = render_scene(snapshot, nav, size, envelopes=False)
            scene_cache["key"] = key
        for method, variable in (dashboard.path_status or {}).items():
            entry = next((e for e in snapshot["paths"] if e["method"] == method), None)
            suffix = (
                " …"
                if entry and entry.get("stale")
                else " ×"
                if entry and not entry["valid"]
                else ""
            )
            variable.set(
                (
                    "iMF"
                    if method == "imf"
                    else method.upper()
                    if method != "direct"
                    else "Direct"
                )
                + suffix
            )
        return scene_cache["image"]

    from .ui.scene_inspector import SceneNavigation

    def pick_scene(event):
        if (
            state["motion_job"] is not None
            or state["robot_io_paused"]
            or state.get("calibration_job") is not None
        ):
            status.set(
                "Wait for motion to finish or press Stop before selecting another target"
            )
            return
        if calibration_flow.stage is not None:
            return
        try:
            from .test_bed import table_height_at
            from .ui.scene_inspector import pick_in_workspace, pick_on_plane

            size = (max(160, right.winfo_width()), max(120, right.winfo_height()))
            z = config["table_z_m"] + (
                0 if state.get("discard_points") is not None else config["hover_m"]
            )
            picker = (
                pick_on_plane
                if state.get("discard_points") is not None
                else pick_in_workspace
            )
            point = picker(
                scene_snapshot(), scene_nav.view, size, (event.x, event.y), z
            )
            source = point.copy()
            source[2] = float(table_height_at(source[:2], config))
            if config.get("test_bed"):
                source[2] -= config["test_bed"].get("clearance_margin_m", 0.003)
            state["selection_in_3d"] = True
            accept_world_selection(source, requested_hover=float(point[2] - source[2]))
        except (ValueError, RuntimeError) as exc:
            status.set(f"3D target rejected: {exc}")

    def scene_handle(phase, event):
        """Drag workspace bounds in the 3D wireframe view (no robot motion)."""
        from .ui.scene_inspector import bounds_handle_at, drag_bounds

        size = (max(160, right.winfo_width()), max(120, right.winfo_height()))
        if phase == "press":
            if (
                not config.get("show_workspace", True)
                or calibration_flow.stage is not None
                or state["motion_job"] is not None
                or state["robot_io_paused"]
            ):
                return False
            handle = bounds_handle_at(
                scene_snapshot(), scene_nav.view, size, (event.x, event.y)
            )
            if handle is None:
                return False
            state["scene_bounds_drag"] = (
                copy.deepcopy(config),
                handle,
                (event.x, event.y),
            )
            scene_nav.view.freeze_framing = True
            state["plan"] = None
            if go_button:
                go_button.configure(state="disabled")
            right.configure(cursor="fleur")
            status.set(
                f"Drag {handle[0]} · release to save · Esc to cancel. No robot motion."
            )
            return True
        drag = state.get("scene_bounds_drag")
        if drag is None:
            return False
        original, handle, press = drag
        if phase == "drag":
            try:
                low, high = drag_bounds(
                    scene_snapshot(),
                    scene_nav.view,
                    size,
                    handle,
                    press,
                    (event.x, event.y),
                    original,
                )
                config.update(workspace_min_m=low, workspace_max_m=high)
                lo, hi = np.asarray(low), np.asarray(high)
                status.set(
                    f"Bounds {np.round(lo * 100, 1)} → {np.round(hi * 100, 1)} cm"
                )
                show(right, diagram(state["q"], plan=None))
            except ValueError as exc:
                status.set(str(exc))
            return True
        state.pop("scene_bounds_drag", None)
        scene_nav.view.freeze_framing = False
        right.configure(cursor="")
        if np.linalg.norm(np.asarray([event.x, event.y]) - press) < 4:
            # A click near a handle is still a target selection. Only an
            # actual drag edits bounds, especially in a compact viewport.
            config.update(original)
            pick_scene(event)
            return True
        try:
            if config_path:
                from .calibration_session import write_json

                write_json(config_path, config)
            state["surface_point_m"] = None
            state["selected_pixel"] = None
            status.set("Bounds saved. Select a fresh target; Go starts motion.")
        except OSError as exc:
            config.update(original)
            status.set(f"Could not save bounds; edit reverted: {exc}")
        return True

    scene_nav = SceneNavigation(
        right,
        lambda: show(right, diagram(state["q"], plan=state.get("plan"))),
        on_pick=pick_scene,
        on_hover=lambda event: wheel(event),
        on_handle=scene_handle,
    )

    def animate(index=0):
        plan = state["plan"]
        if plan is None:
            return
        state["q"] = plan.joints_rad[index]
        show(right, diagram(state["q"], plan.target_m, plan))
        if index + 1 < len(plan.time_s):
            delay = max(
                1,
                int(
                    1000
                    * (plan.time_s[index + 1] - plan.time_s[index])
                    / (4 if passive else 1)
                ),
            )
            state["anim"] = root.after(delay, animate, index + 1)
        else:
            state["anim"] = None
            if not passive:
                clear_paths()

    def clear_paths():
        nonlocal route_version
        route_version += 1
        state["plan"] = None
        state["path_overlays"] = []
        state["surface_point_m"] = None
        for key in ("overlay_future",):
            if state.get(key) is not None:
                state[key].cancel()
        if state.get("anim") is not None:
            root.after_cancel(state["anim"])
            state["anim"] = None
        state["selection_executed"] = True
        if go_button:
            go_button.configure(state="disabled")
        show(right, diagram(state["q"]))

    def poll_motion():
        job = state["motion_job"]
        if job is None:
            return
        code = job.process.poll()
        if code is None:
            root.after(100, poll_motion)
            return
        clear_paths()
        try:
            result = job.finish()
            if code == 0 and result and result.get("stopped"):
                status.set("Held · " + result["reason"])
                state["motion_job"] = None
                state["selection_executed"] = True
                go_button.configure(state="disabled")
                if state["close_requested"]:
                    root.destroy()
                return
            if code == 0 and result and result.get("completed"):
                if result.get("can_discard_completed"):
                    status.set(
                        f"Can followed in RGB · {result.get('can_attempts', 1)} attempt(s) · release commanded in zone; drop unverified"
                    )
                    state["motion_job"] = None
                    return
                if "grasp_confirmed" in result:
                    status.set(
                        (
                            "Keys held · camera confirmed"
                            if result["grasp_confirmed"]
                            else "Grab not confirmed · retry limit reached · holding"
                        )
                        + f" | {result['attempts']} attempts"
                    )
                    state["motion_job"] = None
                    state["selection_executed"] = True
                    go_button.configure(state="disabled")
                    if state["close_requested"]:
                        root.destroy()
                    return
                if result.get("home_completed"):
                    status.set(
                        "Home reached · "
                        + (
                            "gripper closed"
                            if result.get("gripper_closed")
                            else "gripper closure obstructed · holding"
                        )
                    )
                    state["motion_job"] = None
                    state["selection_executed"] = True
                    go_button.configure(state="disabled")
                    if state["close_requested"]:
                        root.destroy()
                    return
                gap = result["settled_model_endpoint_error_mm"]
                reached = result.get(
                    "model_target_reached", gap <= config["goal_tolerance_m"] * 1000
                )
                status.set(
                    f"Motion finished. Estimated tip gap {gap:.1f} mm — "
                    + (
                        "within target tolerance. "
                        if reached
                        else "target NOT reached within tolerance. "
                    )
                    + "Holding. "
                    + planner_status(
                        result.get("planner_info"), config.get("planner_mode")
                    )
                )
            else:
                details = (
                    read_process_log(job.directory / "stderr.log").strip().splitlines()
                )
                status.set(
                    "Motion did not complete. "
                    + (
                        details[-1]
                        if details
                        else "Check the motor status before another trial."
                    )
                )
        except (OSError, ValueError, KeyError) as exc:
            status.set(f"Motion process ended; report unavailable: {exc}")
        state["motion_job"] = None
        state["selection_executed"] = True
        go_button.configure(state="disabled")
        if state["close_requested"]:
            root.destroy()

    def go():
        if (
            not passive
            or state.get("bounds_drag") is not None
            or state.get("hold_pending_clear", False)
            or state["plan"] is None
            or state["motion_job"] is not None
            or state["selection_executed"]
        ):
            return
        if state.get("comparison_open") or (
            config["trajectory_method"] == "direct"
            and state.get("obstacle_boxes")
            and not config.get("allow_object_collisions")
        ):
            status.set(
                "Close Compare, or select an obstacle-aware planner in the sidebar. Direct does not avoid objects."
            )
            return
        try:
            if (
                state["robot_io_paused"]
                or state["motor"] is None
                or time.monotonic() - state["last_state_read"] > 0.5
            ):
                raise ValueError("Wait for fresh joint readings before Go")
            if state["frame"] is None or state["surface_point_m"] is None:
                raise ValueError("Select a target in the 3D workspace")
            # The selected world target is fixed. Camera occlusions or missing
            # depth do not invalidate it; motion uses encoder feedback.
            if state["anim"] is not None:
                root.after_cancel(state["anim"])
                state["anim"] = None
            if route_info(state["plan"]):
                require_same_scene(
                    route_info(state["plan"]),
                    {
                        "unix_time_s": time.time(),
                        "camera_stale": state.get("camera_stale"),
                        "obstacle_detection": state.get("obstacle_report"),
                        "obstacle_boxes": [
                            b.as_dict() for b in state.get("obstacle_boxes", [])
                        ],
                    },
                    arm=arm,
                    joints=state["plan"].joints_rad,
                )
            job_config = copy.deepcopy(config)
            source = state.get("auto_grasp_source")
            if source is not None or config.get("click_and_grab"):
                job_config["grasp_request"] = dict(
                    source_m=source
                    if source is not None
                    else state["surface_point_m"].tolist(),
                    max_retries=5,
                    reference_pixel=state.get("auto_grasp_reference_pixel"),
                )
            else:
                job_config.pop("grasp_request", None)
            state["motion_job"] = launch(state["plan"], job_config, state["motor"])
            go_button.configure(state="disabled")
            status.set("Go requested · executing reach; Stop retains hold.")
            root.after(100, poll_motion)
        except (OSError, ValueError, RuntimeError) as exc:
            go_button.configure(state="disabled")
            status.set(f"Go blocked: {exc}")

    def close_when_idle():
        if (
            state["motion_job"] is None
            and state["calibration_job"] is None
            and not Path("runs/control/pause_robot").exists()
        ):
            root.destroy()
        else:
            root.after(100, close_when_idle)

    def close_window():
        if (
            state["motion_job"] is not None
            or state["robot_io_paused"]
            or state["calibration_job"] is not None
        ):
            request_stop()
            state["close_requested"] = True
            root.after(100, close_when_idle)
        else:
            root.destroy()

    def close_with_recording():
        if recorder.recording:
            recorder.stop()
        close_window()

    root.protocol("WM_DELETE_WINDOW", close_with_recording)

    def change_grab_mode(value):
        config["click_and_grab"] = bool(value)
        change_hover(config["hover_m"])

    def preview_point(
        point, pixel=None, depth=None, for_grasp=False, requested_hover=None
    ):
        nonlocal route_future, route_version
        route_version += 1
        if route_future is not None:
            route_future.cancel()
        point = np.asarray(point)
        if state.get("selection_in_3d"):
            minimum = max(0.0, config["workspace_min_m"][2] - point[2])
            maximum = config["workspace_max_m"][2] - point[2]
            if maximum < minimum:
                raise ValueError("No target height inside the configured workspace")
        else:
            minimum, maximum = hover_bounds(point, config)
        if (
            state["frame"] is not None
            and config["calibration_verified"]
            and not state.get("selection_in_3d")
        ):
            from .overlays import visible_hover_bounds

            rgb, _, intrinsics = state["frame"]
            minimum, maximum = visible_hover_bounds(
                point,
                (minimum, maximum),
                config["T_base_camera"],
                intrinsics,
                (rgb.shape[1], rgb.shape[0]),
            )
        dashboard.hover.set_limits(minimum, maximum)
        if requested_hover is not None:
            dashboard.hover.set_value(requested_hover, notify=False)
        config["hover_m"] = dashboard.hover.value
        state["surface_point_m"] = point.copy()
        state["selected_pixel"] = pixel
        state["selected_depth"] = depth
        state["selection_executed"] = False
        state["plan"] = None
        state["path_overlays"] = [
            {**entry, "stale": True} for entry in state["path_overlays"]
        ]
        # Keep the last comparison visible until its replacement is ready.
        if go_button:
            go_button.configure(state="disabled")
        if state["anim"] is not None:
            root.after_cancel(state["anim"])
            state["anim"] = None
        if passive and state["motor"] is None:
            raise ValueError("Wait for fresh motor feedback")

        target = point + [0, 0, config["hover_m"]]
        started = time.perf_counter()
        if passive and not config["joint_mapping_verified"]:
            raise ValueError(
                "Model signs and offsets need verification before live-state planning"
            )
        planning_config = dict(config)
        if passive:
            planning_config["max_joint_speed_deg_s"] = min(
                HARDWARE_SPEED_LIMIT_DEG_S, config["max_joint_speed_deg_s"]
            )
        seed = state["q"]
        version = route_version
        prior_overlay = state.get("overlay_future")
        if prior_overlay is not None:
            prior_overlay.cancel()
        overlay_cfg = copy.deepcopy(planning_config)
        overlay_boxes = copy.deepcopy(state.get("obstacle_boxes", []))
        overlay_seed = np.array(seed).copy()

        import queue

        overlay_updates = queue.SimpleQueue()

        def compare_paths():
            if version != route_version:
                return []
            from .trajectory.cancel import planning_scope

            with planning_scope(lambda: version != route_version):
                return overlay_planner.compare(
                    arm,
                    overlay_seed,
                    target.copy(),
                    overlay_cfg,
                    overlay_boxes,
                    overlay_cfg["trajectory_candidates"],
                    (state.get("obstacle_report") or {}).get("scene_revision", 0),
                    on_update=overlay_updates.put,
                )

        future_overlays = overlay_executor.submit(compare_paths)
        state["overlay_future"] = future_overlays

        def overlay_done():
            if version != route_version:
                return
            while not overlay_updates.empty():
                update = overlay_updates.get()
                methods = {entry["method"] for entry in update}
                state["path_overlays"] = update + [
                    entry
                    for entry in state["path_overlays"]
                    if entry["method"] not in methods
                ]
            if not future_overlays.done():
                root.after(100, overlay_done)
                return
            try:
                state["path_overlays"] = future_overlays.result()
            except (ValueError, RuntimeError, OSError) as exc:
                status.set(f"Comparison refresh unavailable: {exc}")

        root.after(100, overlay_done)
        want_grasp = for_grasp or config.get("click_and_grab", False)

        def with_grasp_pose(plan):
            if want_grasp:
                from .grasp_geometry import attach_grasp_preview

                return attach_grasp_preview(plan, arm, planning_config, point)
            return plan

        if config["trajectory_method"] != "direct" or config.get(
            "allow_object_collisions"
        ):
            report = state.get("obstacle_report") or {}
            if live and (
                report.get("error")
                or (
                    report.get("change_pending")
                    and not config.get("allow_object_collisions")
                )
                or "scene_revision" not in report
            ):
                raise ValueError("Wait for stable object detection before planning")
            captured_config = copy.deepcopy(planning_config)
            captured_boxes = copy.deepcopy(state.get("obstacle_boxes", []))
            version = route_version
            captured_seed = np.array(seed).copy()
            method, count = config["trajectory_method"], config["trajectory_candidates"]
            revision = report.get("scene_revision", 0)

            def compute_preview():
                from .trajectory.cancel import planning_scope

                with planning_scope(lambda: version != route_version):
                    return with_grasp_pose(
                        route_planner.plan(
                            arm,
                            captured_seed,
                            target.copy(),
                            captured_config,
                            captured_boxes,
                            method,
                            count,
                            revision,
                        )
                    )

            future = route_executor.submit(compute_preview)
            route_future = future
            status.set(f"Planning {config['trajectory_method'].upper()}…")

            def finished():
                if version != route_version:
                    return
                if not future.done():
                    root.after(75, finished)
                    return
                try:
                    plan = future.result()
                    if state["surface_point_m"] is None or not np.allclose(
                        state["surface_point_m"], point
                    ):
                        return
                    if live:
                        require_same_scene(
                            route_info(plan),
                            dict(
                                unix_time_s=time.time(),
                                camera_stale=state.get("camera_stale"),
                                obstacle_detection=state.get("obstacle_report"),
                                obstacle_boxes=[
                                    b.as_dict() for b in state.get("obstacle_boxes", [])
                                ],
                            ),
                            arm=arm,
                            joints=plan.joints_rad,
                        )
                    finish_preview(plan, point, pixel, depth, target, started)
                except (ValueError, RuntimeError, OSError) as exc:
                    state["plan"] = None
                    status.set(str(exc))

            root.after(75, finished)
            return
        if passive and state["motor"] is not None:
            plan = preview_from_feedback(
                arm, seed, state["motor"], target, planning_config
            )
        else:
            plan = plan_reach(arm, seed, target, planning_config)
        finish_preview(with_grasp_pose(plan), point, pixel, depth, target, started)

    def finish_preview(plan, point, pixel, depth, target, started):
        elapsed = (time.perf_counter() - started) * 1000
        state["plan"] = plan
        state["surface_point_m"] = np.asarray(point)
        save_plan(Path("runs/last_plan.json"), plan, config, synthetic=not live)
        if passive:
            state["selected_pixel"] = pixel
            state["selected_depth"] = depth
            ready = arm_holding(state["motor"])
            fault = tracking_fault(state["motor"])
            if motion_disabled():
                suffix = "Disabled: preview only. Enable allows new commands."
            elif state.get("hold_pending_clear", False):
                suffix = "Hold on: confirm support and hands clear below."
            elif state.get("comparison_open") or (
                config["trajectory_method"] == "direct"
                and state.get("obstacle_boxes")
                and not config.get("allow_object_collisions")
            ):
                suffix = "Direct blocked by objects · select RRT, MLP, FM, MF or iMF."
            elif ready and not fault:
                go_button.configure(state="normal")
                suffix = "Press Go to execute."
            elif fault:
                suffix = "Go blocked: " + fault
            else:
                suffix = "Motors off · press Enable."
            if plan.recovery_waypoint_m is not None:
                suffix = "Includes a waypoint to enter the selection region. " + suffix
        else:
            suffix = "OFFLINE PREVIEW"
        suffix += " " + planner_status(plan.planner_info, config.get("planner_mode"))
        grasp = next((i for i in plan.planner_info or [] if "tilt_deg" in i), None)
        if grasp:
            suffix += f" Grab tilt {grasp['tilt_deg']:.1f}° from table normal."
            if grasp.get("table_clearance_m", 0) < 0:
                suffix += f" Planned contact: {grasp['closest_table_body']} table clearance {grasp['table_clearance_m'] * 1000:.1f} mm."
                if not config.get("allow_object_collisions"):
                    if go_button:
                        go_button.configure(state="disabled")
                    suffix += " Go blocked at descent preview."
                else:
                    suffix += " Collision override ON."
        trajectory = route_info(plan)
        route_label = ""
        if trajectory:
            route_label = trajectory["trajectory_method"].upper()
            if trajectory.get("joint_space_repair"):
                route_label += " + whole-arm RRT repair (fallback)"
            refinement = trajectory.get("path_refinement", {})
            if refinement.get("applied"):
                route_label += f" + clearance refinement ({refinement['max_shift_m'] * 100:.1f} cm)"
            route_label += " | "
            if trajectory.get("allow_object_collisions"):
                suffix = "Collision override ON · " + suffix
                if trajectory.get("collision_predictions") or trajectory.get(
                    "tool_route_collision"
                ):
                    suffix = "Predicted object collision · " + suffix
        status.set(
            route_label
            + f"Target {np.round(target, 3)} m | model error {plan.endpoint_error_m * 1000:.2f} mm | "
            f"plan {elapsed:.0f} ms | duration {plan.time_s[-1]:.1f} s | "
            + (
                state.get("selection_source", {}).get("source", "") + " | "
                if live
                else ""
            )
            + suffix
        )
        if passive:
            # The live arm diagram always uses measured encoders. Clicking only
            # changes the red planned path and target, not the displayed arm pose.
            state["q"] = np.array(state["motor"]["q_model_rad"])
            show(right, diagram(state["q"], plan.target_m, plan))
        else:
            animate()

    def change_trajectory(method, count):
        config["trajectory_method"] = method
        config["trajectory_candidates"] = count
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set("Planner selection applies to the next reach.")
            return
        change_hover(config["hover_m"])

    def change_planner(value):
        if state["motion_job"] is not None or state["robot_io_paused"]:
            status.set(
                "Planner changes apply after the current motor operation finishes."
            )
            return False
        config["planner_mode"] = value
        status.set(
            "Planner: "
            + (
                "Flow proposal + numerical refinement"
                if value == "flow_assisted"
                else "Numerical IK"
            )
        )
        change_hover(config["hover_m"])
        return True

    def change_hover(value):
        state.pop("auto_grab_token", None)
        if state["motion_job"] is not None or state["robot_io_paused"]:
            dashboard.hover.set_value(config["hover_m"], notify=False)
            return
        config["hover_m"] = float(value)
        state["plan"] = None
        if go_button:
            go_button.configure(state="disabled")
        if state.get("hover_preview_job") is not None:
            root.after_cancel(state["hover_preview_job"])

        def update():
            state["hover_preview_job"] = None
            try:
                if state["surface_point_m"] is not None:
                    preview_point(
                        state["surface_point_m"],
                        state["selected_pixel"],
                        state["selected_depth"],
                    )
                if config_path:
                    from .calibration_session import write_json

                    write_json(config_path, config)
            except (ValueError, RuntimeError, OSError) as exc:
                status.set(f"Hover {config['hover_m'] * 100:.1f} cm: {exc}")

        state["hover_preview_job"] = root.after(120, update)

    def click(event):
        if state["robot_io_paused"] or state["motion_job"] is not None:
            status.set(
                "Motor connection is with the motion controller. Use Stop motion (hold) to stop the reach."
            )
            return
        if state["anim"] is not None:
            root.after_cancel(state["anim"])
            state["anim"] = None
        if passive and state["motor"] is not None:
            state["q"] = np.array(state["motor"]["q_model_rad"])
        state["plan"] = None
        state["selected_pixel"] = None
        state["surface_point_m"] = None
        state["selection_executed"] = False
        if go_button:
            go_button.configure(state="disabled")
        viewport = getattr(
            event.widget if hasattr(event, "widget") else left,
            "viewport",
            left.viewport,
        )
        pixel = viewport.to_camera(event.x, event.y)
        if pixel is None:
            return
        event.x, event.y = pixel
        try:
            if camera:
                if (
                    config["T_base_camera"] is None
                    or not config["calibration_verified"]
                ):
                    calibrated = False
                else:
                    calibrated = True
                if state["frame"] is None:
                    raise ValueError("Wait for camera frames")
                if passive and (
                    state["motor"] is None
                    or time.monotonic() - state["last_state_read"] > 0.5
                ):
                    raise ValueError(
                        "Joint readings are unavailable or stale; wait for them to recover"
                    )
                _, depth, intrinsics = state["frame"]
                if calibration_flow.stage == "paper":
                    calibration_flow.capture_paper((event.x, event.y))
                    return
                if calibration_flow.stage == "paper_webcam":
                    status.set("Click the four A4 corners in the webcam view first.")
                    return
                if calibration_flow.stage in ("release", "hold", "clear"):
                    status.set(
                        "Complete the current calibration step below the camera views."
                    )
                    return
                if calibrated:
                    point, source = scene_map.point(
                        event.x, event.y, intrinsics, config
                    )
                    state["selection_source"] = source
                    p_camera = transform_point(
                        np.linalg.inv(np.asarray(config["T_base_camera"])), point
                    )
                else:
                    p_camera = camera.point(depth, intrinsics, event.x, event.y)
                if passive and not calibrated:
                    if state.get("camera_stale"):
                        raise ValueError(
                            "Calibration captures require fresh camera frames"
                        )
                    if len(state["tool_points"]) >= 10:
                        assess_if_ready()
                        return
                    distance, depth_quality = calibration_depth(
                        state["depth_history"], event.x, event.y
                    )
                    p_camera = camera.deproject(intrinsics, event.x, event.y, distance)
                    motor = state["motor"]
                    if motor is None:
                        raise ValueError("Wait for joint readings")
                    if any(motor["torque_enabled"].values()):
                        raise ValueError(
                            "Manual capture requires every motor's torque to be disabled"
                        )
                    if motor["gripper_percent"] > 8:
                        raise ValueError(
                            "Gently close the gripper jaws by hand before clicking their meeting point"
                        )
                    recent = [
                        item[1]
                        for item in state["history"]
                        if time.monotonic() - item[0] < 0.7
                    ]
                    if (
                        len(recent) < 3
                        or np.rad2deg(np.ptp(recent, axis=0)).max() > 0.7
                    ):
                        raise ValueError(
                            "Hold the arm still for one second, then click again"
                        )
                    q_capture = np.array(motor["q_model_rad"])
                    if (q_capture < arm.lower).any() or (q_capture > arm.upper).any():
                        raise ValueError(
                            "Move the joints away from their end stops into a comfortable middle pose"
                        )
                    base = arm.fk(q_capture)
                    if any(
                        np.linalg.norm(base - r["base_point_model_m"]) < 0.025
                        for r in state["tool_points"]
                    ):
                        raise ValueError(
                            "Choose a new tool position at least 2.5 cm from earlier samples"
                        )
                    record = {
                        "session_id": state["session_id"],
                        "unix_time_s": time.time(),
                        "pixel": [event.x, event.y],
                        "camera_point_m": p_camera.tolist(),
                        "base_point_model_m": base.tolist(),
                        "q_model_rad": q_capture.tolist(),
                        "camera_serial": config["camera_serial"],
                        "gripper_percent": motor["gripper_percent"],
                        "depth_quality": depth_quality,
                    }
                    count = len(state["tool_points"]) + 1
                    capture_dir = Path("runs/calibration") / state["session_id"]
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    rgb_path = capture_dir / f"point_{count:02d}_rgb.png"
                    depth_path = capture_dir / f"point_{count:02d}_depth_m.npy"
                    marked = Image.fromarray(state["frame"][0])
                    mark = ImageDraw.Draw(marked)
                    x, y = event.x, event.y
                    mark.ellipse(
                        (x - 7, y - 7, x + 7, y + 7), outline="#ff4060", width=2
                    )
                    mark.text(
                        (10, 10),
                        f"Point {count} | depth {distance * 100:.1f} cm",
                        fill="#ff4060",
                    )
                    marked.save(rgb_path)
                    np.save(depth_path, depth.astype(np.float32))
                    record.update(rgb_file=str(rgb_path), depth_file=str(depth_path))
                    with Path("runs/tool_points.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(record) + "\n")
                    state["tool_points"].append(record)
                    remember_session(state["session_id"], config)
                    count = len(state["tool_points"])
                    if count >= 8:
                        assess_if_ready()
                    else:
                        calibration_flow.capture_step()
                        status.set(
                            f"Captured tool pose {count}/8. Move to another position/height; hold still and click the same jaw-tip point."
                        )
                    return
                capture = Path("runs/camera_clicks.jsonl")
                capture.parent.mkdir(parents=True, exist_ok=True)
                with capture.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "unix_time_s": time.time(),
                                "pixel": [event.x, event.y],
                                "camera_point_m": p_camera.tolist(),
                                "camera_serial": config["camera_serial"],
                                "motor_state": state["motor"],
                                "tip_tracking": state.get("tip_tracking"),
                            }
                        )
                        + "\n"
                    )
                if not calibrated:
                    status.set(
                        f"Captured camera XYZ {np.round(p_camera, 4)} m in runs/camera_clicks.jsonl. "
                        "Measure matching base XYZ for calibration; planning requires a verified transform."
                    )
                    return
            else:
                point = synthetic_point(event.x, event.y)
            preview_point(
                point, (event.x, event.y), float(p_camera[2])
            ) if camera else preview_point(point)
        except (ValueError, RuntimeError, OSError) as exc:
            status.set(f"Rejected: {exc}")

    def refresh():
        if camera:
            if (live_dir / "stop").exists():
                root.destroy()
                return
            try:
                from PIL import Image

                try:
                    state["frame"] = camera.read()
                    state["camera_stale"] = False
                    scene_map.update(state["frame"][1])
                except (OSError, RuntimeError, ValueError):
                    if state["frame"] is None:
                        raise
                    state["camera_stale"] = True
                rgb, depth, intrinsics = state["frame"]
                if webcam is not None:
                    state["secondary_frame"] = webcam.read()
                    if (
                        state.get("tip_tracker") is not None
                        and state["secondary_frame"] is not None
                    ):
                        webcam_image, webcam_stamp = state["secondary_frame"]
                        was_tracked = (state.get("tip_tracking") or {}).get("valid")
                        state["tip_tracking"] = state["tip_tracker"].update(
                            webcam_image, webcam_stamp
                        )
                        if was_tracked and not state["tip_tracking"]["valid"]:
                            status.set(
                                "Webcam tip tracking lost. Select tip again; no colour-based replacement was chosen."
                            )
                if passive:
                    paused = Path("runs/control/pause_robot").exists()
                    if paused and not state["robot_io_paused"]:
                        passive.close()
                        state["robot_io_paused"] = True
                        state["motor"] = None
                        state["history"] = []
                        torque_status.set(
                            "Motor port handed to the motion controller. Camera remains live. Stop motion retains hold."
                        )
                    elif not paused and state["robot_io_paused"]:
                        passive.resume()
                        state["robot_io_paused"] = False
                        state["last_state_read"] = 0.0
                state["depth_history"] = (state["depth_history"] + [state["frame"][1]])[
                    -5:
                ]
                preview = Image.fromarray(state["frame"][0])
                if state.get("camera_stale"):
                    ImageDraw.Draw(preview).text(
                        (12, preview.height - 22),
                        "CAMERA STALE — using saved scene estimate",
                        fill="#ffcf83",
                    )
                if passive and not config["calibration_verified"]:
                    ImageDraw.Draw(preview).rectangle(
                        (40, 40, 599, 439), outline="#75d6b3", width=1
                    )
                if passive and state["robot_io_paused"]:
                    try:
                        control = json.loads(
                            Path("runs/control/state.json").read_text(encoding="utf-8")
                        )
                        if time.time() - control["unix_time_s"] < 0.5 and control.get(
                            "motor_state"
                        ):
                            state["motor"] = control["motor_state"]
                            state["q"] = np.array(state["motor"]["q_model_rad"])
                    except (OSError, ValueError, KeyError):
                        pass
                measured = state["motor"]["q_model_rad"] if state["motor"] else None
                if config["calibration_verified"] and not state.get("camera_stale"):
                    try:
                        source = (
                            scene_map.depth if scene_map.depth is not None else depth
                        )
                        secondary_args = (
                            (state["secondary_frame"],)
                            if state["secondary_frame"] is not None
                            else ()
                        )
                        planning = any(
                            future is not None and not future.done()
                            for future in (state.get("overlay_future"), route_future)
                        )
                        object_detector.min_interval_s = 0.2 if planning else 0.05
                        result = object_detector.update(
                            rgb,
                            source,
                            intrinsics,
                            config,
                            arm,
                            state["q"] if measured is None and passive else measured,
                            *secondary_args,
                        )
                        if result is not None:
                            state["obstacle_boxes"], state["obstacle_report"] = result
                            existing = state.get("plan")
                            info = route_info(existing) if existing else None
                            if (
                                info
                                and state["motion_job"] is None
                                and info["scene_revision"]
                                != result[1].get("scene_revision")
                            ):
                                state["plan"] = None
                                if go_button:
                                    go_button.configure(state="disabled")
                                change_hover(config["hover_m"])
                    except (ValueError, RuntimeError, OSError) as exc:
                        state["obstacle_report"] = {"error": str(exc)}
                target = (
                    state["surface_point_m"] + [0, 0, config["hover_m"]]
                    if state["surface_point_m"] is not None
                    else None
                )
                if config["calibration_verified"]:
                    state["visual_check"] = draw_overlays(
                        preview,
                        arm,
                        measured,
                        target,
                        state["surface_point_m"],
                        config,
                        intrinsics,
                        plan=None,
                    )
                    from .trajectory.perception import draw_boxes

                    draw_boxes(
                        preview, state.get("obstacle_boxes", []), config, intrinsics
                    )
                    key_draw = ImageDraw.Draw(preview)
                    for key in (state.get("obstacle_report") or {}).get(
                        "key_detections", []
                    ):
                        key_draw.rectangle(key["pixel_roi"], outline="#77cbe8", width=2)
                        if key.get("grasp_pixel"):
                            gx, gy = key["grasp_pixel"]
                            key_draw.ellipse(
                                (gx - 4, gy - 4, gx + 4, gy + 4),
                                outline="#77ff99",
                                width=2,
                            )

                        x, y = key["pixel_roi"][:2]
                        key_draw.text(
                            (x, max(0, y - 15)),
                            f"Keys {key['confidence']:.0%}",
                            fill="#b6eaff",
                            stroke_width=1,
                            stroke_fill="#171719",
                        )

                can_boxes = state.get("obstacle_boxes", [])
                hole = current_hole()
                if config["calibration_verified"]:
                    from .overlays import draw_can_estimates, view_project

                    draw_can_estimates(
                        preview,
                        lambda p: view_project(p, config, intrinsics),
                        can_boxes,
                        hole,
                    )
                if calibration_flow.stage is not None and calibration_flow.pixels:
                    draw = ImageDraw.Draw(preview)
                    pixels = calibration_flow.pixels
                    if len(pixels) == 4:
                        # The sheet exactly where the corners were placed.
                        outline = [tuple(p) for p in pixels]
                        draw.line(outline + [outline[0]], fill="#68f1a6", width=2)
                    for index, (x, y) in enumerate(pixels, 1):
                        draw.ellipse(
                            (x - 5, y - 5, x + 5, y + 5), outline="#ff9a70", width=2
                        )
                        draw.text(
                            (x + 8, y - 8),
                            str(index),
                            fill="white",
                            stroke_width=1,
                            stroke_fill="black",
                        )
                if config["calibration_verified"]:
                    from .overlays import view_project

                    def projection(p):
                        return view_project(p, config, intrinsics)

                show(left, preview)
                depth_color, gray_depth = depth_images(
                    scene_map.depth if scene_map.depth is not None else depth
                )
                state["gray_depth"] = gray_depth
                if config["calibration_verified"]:
                    draw_overlays(
                        depth_color,
                        arm,
                        measured,
                        target,
                        state["surface_point_m"],
                        config,
                        intrinsics,
                        plan=None,
                    )
                    draw_boxes(
                        depth_color, state.get("obstacle_boxes", []), config, intrinsics
                    )
                    draw_can_estimates(
                        depth_color,
                        lambda p: view_project(p, config, intrinsics),
                        can_boxes,
                        hole,
                    )
                else:
                    ImageDraw.Draw(depth_color).text(
                        (12, 12), "DEPTH / 0.2–1.0 m / missing: black", fill="white"
                    )
                show(depth_view, depth_color)
                if state["secondary_frame"] is not None:
                    second_rgb, stamp = state["secondary_frame"]
                    second_image = Image.fromarray(second_rgb)
                    second_draw = ImageDraw.Draw(second_image)
                    from .multiview import calibration_current, project_secondary

                    calibration = config.get("secondary_camera_calibration")
                    if calibration_current(
                        config, (second_rgb.shape[1], second_rgb.shape[0])
                    ) and not (state.get("obstacle_report") or {}).get(
                        "secondary_view", {}
                    ).get("alignment_stale"):
                        second_draw.line(
                            [tuple(p) for p in calibration["secondary_pixels"]]
                            + [tuple(calibration["secondary_pixels"][0])],
                            fill="#63D7C4",
                            width=2,
                        )
                        if calibration.get("T_base_camera") and calibration.get(
                            "intrinsics"
                        ):
                            from .workspace_geometry import box_edges

                            for number, box in enumerate(
                                state.get("obstacle_boxes", []), 1
                            ):
                                edges = box_edges(
                                    dict(
                                        workspace_min_m=box.low,
                                        workspace_max_m=box.high,
                                    )
                                )
                                pixels = []
                                for a, b in edges:
                                    pa = project_secondary(
                                        a, calibration, allow_planar=True
                                    )
                                    pb = project_secondary(
                                        b, calibration, allow_planar=True
                                    )
                                    if pa is not None and pb is not None:
                                        second_draw.line(
                                            (pa, pb), fill="#ffba75", width=2
                                        )
                                        pixels.extend([pa, pb])
                                if pixels:
                                    x, y = np.min(pixels, axis=0)
                                    second_draw.text(
                                        (x, y - 15),
                                        f"Box {number}",
                                        fill="#ffba75",
                                        stroke_width=1,
                                        stroke_fill="#171719",
                                    )
                        if target is not None and calibration.get("T_base_camera"):
                            marker = project_secondary(
                                target, calibration, allow_planar=True
                            )
                            if marker is not None:
                                x, y = marker
                                second_draw.ellipse(
                                    (x - 7, y - 7, x + 7, y + 7),
                                    outline="#63D7C4",
                                    width=3,
                                )
                        if calibration.get("T_base_camera") and calibration.get(
                            "intrinsics"
                        ):
                            from .overlays import draw_can_estimates
                            from .test_bed import corners as paper_corners
                            from .trajectory.rgb_can_perception import projector

                            second_project = projector(
                                calibration["T_base_camera"],
                                calibration["intrinsics"],
                                paper_corners(config, inset=False),
                                calibration["secondary_pixels"],
                            )

                            def webcam_pixel(point):
                                pixel = second_project(np.asarray(point, float)[None])[
                                    0
                                ]
                                return (
                                    tuple(float(v) for v in pixel)
                                    if np.isfinite(pixel).all()
                                    else None
                                )

                            draw_can_estimates(
                                second_image, webcam_pixel, can_boxes, hole
                            )
                        if not calibration.get("metric_3d_validated"):
                            second_draw.text(
                                (12, 12),
                                "A4 alignment · raised pose unverified",
                                fill="#F4C66A",
                                stroke_width=1,
                                stroke_fill="#171719",
                            )
                    else:
                        second_draw.text(
                            (12, 12),
                            "Calibrate views to enable cross-view confirmation",
                            fill="white",
                            stroke_width=1,
                            stroke_fill="#171719",
                        )
                    # Display the shared world-object IDs, not unmatched raw
                    # webcam rectangles that look like separate scene objects.
                    tracked = state.get("tip_tracking") or {}
                    if tracked.get("valid"):
                        tx, ty = tracked["pixel"]
                        drawing = ImageDraw.Draw(second_image)
                        drawing.ellipse(
                            (tx - 7, ty - 7, tx + 7, ty + 7), outline="#68f1a6", width=2
                        )
                        drawing.text((tx + 10, ty), "Tracked tip", fill="#68f1a6")
                    show(webcam_view, second_image)
                elif webcam is not None:
                    webcam_view.configure(
                        image="", text=webcam.error or "Waiting for webcam"
                    )
                if passive and state["cursor"] is not None:
                    x, y = state["cursor"]
                    raw = Image.fromarray(state["frame"][0])
                    crop = raw.crop((x - 12, y - 12, x + 12, y + 12)).resize(
                        (144, 144), Image.Resampling.NEAREST
                    )
                    draw = ImageDraw.Draw(crop)
                    draw.line((65, 72, 79, 72), fill="#ff4060")
                    draw.line((72, 65, 72, 79), fill="#ff4060")
                    show(zoom, crop)
                    gray_crop = (
                        depth_color.crop((x - 12, y - 12, x + 12, y + 12))
                        .resize((144, 144), Image.Resampling.NEAREST)
                        .convert("RGB")
                    )
                    gray_draw = ImageDraw.Draw(gray_crop)
                    gray_draw.line((65, 72, 79, 72), fill="#ff4060")
                    gray_draw.line((72, 65, 72, 79), fill="#ff4060")
                    show(gray_zoom, gray_crop)
                    try:
                        if calibration_mode:
                            distance, _ = calibration_depth(
                                state["depth_history"], x, y
                            )
                            hint = "Click the exact meeting point of the CLOSED jaw tips.\nUse varied positions and heights."
                        else:
                            point, source = scene_map.point(x, y, intrinsics, config)
                            camera_point = transform_point(
                                np.linalg.inv(np.asarray(config["T_base_camera"])),
                                point,
                            )
                            distance = camera_point[2]
                            age = (
                                f" ({source['age_s']:.1f}s old)"
                                if source.get("age_s") is not None
                                else ""
                            )
                            hint = (
                                source["source"]
                                + age
                                + f"\nHover: {config['hover_m'] * 100:.0f} cm"
                            )
                        cursor_info.set(
                            f"Pixel {x}, {y}\nDepth: {distance * 100:.1f} cm\n" + hint
                        )
                    except ValueError as exc:
                        cursor_info.set(str(exc))
                if (
                    passive
                    and not state["robot_io_paused"]
                    and time.monotonic() - state["last_state_read"] > 0.15
                ):
                    state["last_state_read"] = time.monotonic()
                    try:
                        recovered = state.get("motor_error") is not None
                        state["motor"] = passive.read()
                        state["motor_error"] = None
                        torque = state["motor"]["torque_enabled"]
                        count = sum(
                            bool(torque.get(j))
                            for j in (
                                "shoulder_pan",
                                "shoulder_lift",
                                "elbow_flex",
                                "wrist_flex",
                                "wrist_roll",
                            )
                        )
                        torque_status.set(
                            f"Arm hold: {count}/5  |  Gripper: {'on' if torque.get('gripper') else 'off'}"
                        )
                        if "voltage_v" in state["motor"]:
                            state["telemetry"] = state["motor"]
                            sample = state["motor"]["telemetry_unix_s"]
                            if sample > state["last_telemetry_logged"]:
                                try:
                                    with Path("runs/telemetry.jsonl").open(
                                        "a", encoding="utf-8"
                                    ) as stream:
                                        stream.write(
                                            json.dumps(
                                                {
                                                    key: state["motor"][key]
                                                    for key in (
                                                        "telemetry_unix_s",
                                                        "voltage_v",
                                                        "temperature_c",
                                                    )
                                                }
                                            )
                                            + "\n"
                                        )
                                    state["last_telemetry_logged"] = sample
                                except OSError:
                                    pass
                        state["history"].append(
                            (time.monotonic(), np.array(state["motor"]["q_model_rad"]))
                        )
                        state["history"] = state["history"][-12:]
                        if state["anim"] is None:
                            state["q"] = np.array(state["motor"]["q_model_rad"])
                            last_plan = state["plan"]
                            show(
                                right,
                                diagram(
                                    state["q"],
                                    last_plan.target_m if last_plan else None,
                                    last_plan,
                                ),
                            )
                        if recovered:
                            status.set(
                                f"Joint readings recovered. {len(state['tool_points'])}/8 calibration poses captured."
                                if calibration_mode
                                else "Joint readings recovered. Click a clear tabletop point to preview."
                            )
                    except (OSError, RuntimeError, ValueError) as exc:
                        state["motor"] = None
                        state["history"] = []
                        state["motor_error"] = str(exc)
                        torque_status.set(
                            "Motor connection unavailable; current torque state cannot be read."
                        )
                        status.set(
                            "Joint read timed out; retrying. Camera continues; capture is paused until readings recover."
                        )
                if passive and state["robot_io_paused"]:
                    feedback = Path("runs/control/state.json")
                    if feedback.exists():
                        try:
                            control = json.loads(feedback.read_text(encoding="utf-8"))
                            if time.time() - control["unix_time_s"] < 1.0:
                                state["motor"] = control.get("motor_state")
                                if state["motor"]:
                                    state["q"] = np.array(state["motor"]["q_model_rad"])
                                    last_plan = state["plan"]
                                    show(
                                        right,
                                        diagram(
                                            state["q"],
                                            last_plan.target_m if last_plan else None,
                                            last_plan,
                                        ),
                                    )
                                status.set(control["message"])
                        except (OSError, ValueError, KeyError):
                            pass
                if passive and state.get("telemetry"):
                    sample = state["telemetry"]
                    volts = list(sample["voltage_v"].values())
                    temps = list(sample["temperature_c"].values())
                    age = max(0, int(time.time() - sample["telemetry_unix_s"]))
                    telemetry_status.set(
                        f"Supply: {min(volts):.1f}–{max(volts):.1f} V  |  Max temp: {max(temps)} °C"
                        + (f"  ({age}s old)" if age > 4 else "")
                    )
                if passive:
                    fault = tracking_fault(state["motor"])
                    dashboard.fault_label.configure(
                        fg=COLORS.warning if fault else COLORS.muted
                    )
                    if state.get("comparison_open") or (
                        config["trajectory_method"] == "direct"
                        and state.get("obstacle_boxes")
                        and not config.get("allow_object_collisions")
                    ):
                        fault_status.set(
                            "Close Compare or select an obstacle-aware planner in the sidebar"
                        )
                        go_button.configure(state="disabled")
                    elif motion_disabled():
                        fault_status.set(
                            "Disabled · releasing torque; support arm"
                            if state.get("release_pending")
                            else "Disabled · torque still on; retry Disable"
                            if arm_holding(state["motor"])
                            else "Disabled · move by hand; Enable locks current pose"
                        )
                        go_button.configure(state="disabled")
                    elif fault:
                        fault_status.set("Go blocked — " + fault)
                        go_button.configure(state="disabled")
                    elif state["motor"] is None:
                        fault_status.set("Waiting for joint feedback")
                    elif not arm_holding(state["motor"]):
                        fault_status.set("Hold off · preview uses measured joints")
                        go_button.configure(state="disabled")
                    else:
                        actual = state["motor"].get("q_model_rad")
                        goal = state["motor"].get("q_goal_model_rad")
                        maximum = (
                            float(
                                np.rad2deg(
                                    np.abs(np.array(actual) - np.array(goal))
                                ).max()
                            )
                            if actual is not None and goal is not None
                            else 0.0
                        )
                        fault_status.set(
                            f"Holding offset {maximum:.1f}° · tracking stop {TRACKING_LIMIT_DEG:.0f}° · "
                            f"{config['control_hz']:g} Hz requested"
                        )
                    dashboard.disable.configure(
                        text="Releasing…"
                        if state.get("release_pending")
                        else "Disable"
                        if (arm_holding(state["motor"]) or state.get("hold_enabling"))
                        else "Enable"
                    )
                if time.monotonic() - state["last_publish"] > 1.0:
                    rgb, depth, intrinsics = state["frame"]
                    report = {
                        "unix_time_s": time.time(),
                        "motor_state": state["motor"],
                        "tip_tracking": state.get("tip_tracking"),
                        "intrinsics": {
                            "fx": intrinsics.fx,
                            "fy": intrinsics.fy,
                            "ppx": intrinsics.ppx,
                            "ppy": intrinsics.ppy,
                        },
                        "valid_depth_fraction": float((depth > 0).mean()),
                        "camera_serial": config["camera_serial"],
                    }
                    report["calibration_session_id"] = state["session_id"]
                    report["tool_points_captured"] = len(state["tool_points"])
                    report["motor_error"] = state.get("motor_error")
                    report["robot_io_paused"] = state["robot_io_paused"]
                    report["visual_check"] = state["visual_check"]
                    report["scene_model"] = scene_map.summary()
                    report["obstacle_boxes"] = [
                        b.as_dict() for b in state.get("obstacle_boxes", [])
                    ]
                    report["obstacle_detection"] = state.get("obstacle_report")
                    report["planner_overlays"] = state["path_overlays"]
                    report["selection_source"] = state.get("selection_source")
                    report["camera_stale"] = state.get("camera_stale", False)
                    report["secondary_camera"] = dict(
                        name=webcam.name if webcam else None,
                        fresh=state["secondary_frame"] is not None,
                        calibrated=bool(config.get("secondary_camera_calibration")),
                    )
                    try:
                        from .live_snapshot import pickup_images

                        report.update(
                            pickup_images(live_dir, rgb, state["secondary_frame"])
                        )
                        state["snapshot_errors"] = publish(
                            live_dir, np.asarray(preview), depth, report
                        )
                        if state["secondary_frame"] is not None and not state.get(
                            "paired_snapshot_saved"
                        ):
                            import io

                            from .live_snapshot import atomic_bytes

                            folder = Path("runs/deskbuddy/multiview")
                            folder.mkdir(parents=True, exist_ok=True)
                            for name, pixels in [
                                ("primary", rgb),
                                ("secondary", state["secondary_frame"][0]),
                            ]:
                                encoded = io.BytesIO()
                                Image.fromarray(pixels).save(encoded, format="PNG")
                                atomic_bytes(
                                    folder / f"latest-{name}.png", encoded.getvalue()
                                )
                            cloud = io.BytesIO()
                            np.save(cloud, depth)
                            atomic_bytes(folder / "latest-depth.npy", cloud.getvalue())
                            atomic_bytes(
                                folder / "latest-intrinsics.json",
                                json.dumps(report["intrinsics"]).encode(),
                            )
                            state["paired_snapshot_saved"] = True
                        if (
                            config["calibration_verified"]
                            and time.monotonic() - state.get("last_cloud_save", 0) > 5
                        ):
                            import io

                            from .live_snapshot import atomic_bytes

                            cloud = io.BytesIO()
                            np.save(
                                cloud,
                                scene_map.point_cloud(
                                    intrinsics, config["T_base_camera"]
                                ),
                            )
                            atomic_bytes(
                                live_dir / "scene_points_base_m.npy", cloud.getvalue()
                            )
                            state["last_cloud_save"] = time.monotonic()
                    except OSError as exc:
                        state["snapshot_errors"] = {"diagnostics": str(exc)}
                    finally:
                        # Diagnostics failures never invalidate the live frame or stop capture.
                        state["last_publish"] = time.monotonic()
            except (OSError, RuntimeError, ValueError) as exc:
                # Discard the cached frame so clicks cannot use an old image.
                state["frame"] = None
                state["depth_history"] = []
                status.set(f"Camera or snapshot error; retrying: {exc}")
            finally:
                # A transient camera, disk, or motor error must not end the refresh loop.
                show(right, diagram(state["q"], plan=state.get("plan")))
                state["refresh_job"] = root.after(33, refresh)

    def bounds_press(event):
        from .workspace_geometry import box_handles

        if not hasattr(event.widget, "viewport"):
            event.widget = left
        if (
            not config.get("show_workspace", True)
            or state["frame"] is None
            or not config["calibration_verified"]
            or calibration_flow.stage is not None
        ):
            return click(event)
        viewport = event.widget.viewport
        pixel = viewport.to_camera(event.x, event.y)
        if pixel is None:
            return
        intrinsics = state["frame"][2]
        hits = []
        for handle in box_handles(config):
            from .overlays import view_project

            projected = view_project(handle[1], config, intrinsics)
            if projected is not None:
                screen = viewport.from_camera(*projected)
                distance = np.linalg.norm(np.asarray(screen) - [event.x, event.y])
                if distance <= 14:
                    hits.append((distance, handle))
        if not hits:
            return click(event)
        if (
            state["motion_job"] is not None
            or state["calibration_job"] is not None
            or state["robot_io_paused"]
        ):
            status.set("Stop the current operation before dragging bounds.")
            return "break"
        from copy import deepcopy

        handle = min(hits, key=lambda hit: hit[0])[1]
        state["bounds_drag"] = (
            deepcopy(config),
            handle,
            pixel,
            intrinsics,
            event.widget,
        )
        state["plan"] = None
        if go_button:
            go_button.configure(state="disabled")
        if state["anim"] is not None:
            root.after_cancel(state["anim"])
            state["anim"] = None
        event.widget.configure(cursor="fleur")
        if state.get("hover_preview_job") is not None:
            root.after_cancel(state["hover_preview_job"])
            state["hover_preview_job"] = None
        status.set(
            f"Drag {handle[0]} · release to save · Esc to cancel. No robot motion."
        )
        return "break"

    def bounds_motion(event):
        from .workspace_geometry import drag_box

        drag = state.get("bounds_drag")
        if drag is None:
            return
        original, handle, press, intrinsics, widget = drag
        pixel = widget.viewport.to_camera(event.x, event.y)
        if pixel is not None:
            try:
                config.update(drag_box(original, handle, press, pixel, intrinsics))
            except ValueError as exc:
                status.set(str(exc))
        return "break"

    def bounds_release(event=None, cancel=False):
        drag = state.pop("bounds_drag", None)
        if drag is None:
            return
        original, _, _, _, widget = drag
        widget.configure(cursor="")
        if cancel:
            config.update(original)
            status.set("Bounds edit cancelled.")
        else:
            try:
                if config_path:
                    from .calibration_session import write_json

                    write_json(config_path, config)
                state["surface_point_m"] = None
                state["selected_pixel"] = None
                status.set("Bounds saved. Select a fresh target; Go starts motion.")
            except OSError as exc:
                config.update(original)
                status.set(f"Could not save bounds; edit reverted: {exc}")
        return "break"

    for view in (left, depth_view):
        view.bind(
            "<Button-1>",
            lambda event: (
                click(event)
                if calibration_flow.stage is not None or calibration_mode
                else select_camera(event)
            ),
        )
    root.bind("<Escape>", lambda event: bounds_release(cancel=True), add="+")

    def cancel_scene_bounds(event=None):
        drag = state.pop("scene_bounds_drag", None)
        if drag is not None:
            config.update(drag[0])
            scene_nav.view.freeze_framing = False
            scene_nav.handle_drag = False
            right.configure(cursor="")
            status.set("Bounds edit cancelled.")

    root.bind("<Escape>", cancel_scene_bounds, add="+")
    if passive:

        def cursor_move(event):
            state["cursor"] = event.widget.viewport.to_camera(event.x, event.y)

        left.bind("<Motion>", cursor_move)
        depth_view.bind("<Motion>", cursor_move)

    def wheel(event):
        if isinstance(event.widget, tk.Text):
            return
        if (
            state.get("bounds_drag") is not None
            or state["motion_job"] is not None
            or state["robot_io_paused"]
        ):
            return "break"
        delta = getattr(event, "delta", 0)
        if not delta:
            number = getattr(event, "num", 0)
            if number not in (4, 5):
                return
            delta = 120 if number == 4 else -120
        return dashboard.hover.adjust(delta)

    root.bind("<MouseWheel>", wheel)
    root.bind("<Button-4>", wheel)
    root.bind("<Button-5>", wheel)

    def resized(event):
        if event.widget is not root:
            return
        if state.get("resize_job") is not None:
            root.after_cancel(state["resize_job"])

        def redraw():
            state["resize_job"] = None
            if state["frame"] is not None:
                show(left, Image.fromarray(state["frame"][0]))
                show(
                    depth_view,
                    depth_images(
                        scene_map.depth
                        if scene_map.depth is not None
                        else state["frame"][1]
                    )[0],
                )
            elif not camera:
                fixture, depth = synthetic_frame()
                show(left, fixture)
                show(depth_view, depth_images(depth)[0])
            show(
                right,
                diagram(
                    state["q"],
                    state["plan"].target_m if state["plan"] else None,
                    state["plan"],
                ),
            )

        state["resize_job"] = root.after(50, redraw)

    root.bind("<Configure>", resized, add="+")
    show(right, diagram(q))
    show(zoom, Image.new("RGB", (144, 144), COLORS.raised))
    show(gray_zoom, Image.new("RGB", (144, 144), COLORS.raised))
    if camera:
        refresh()
    else:
        fixture, fixture_depth = synthetic_frame()
        show(left, fixture)
        show(depth_view, depth_images(fixture_depth)[0])
    try:
        root.mainloop()
    finally:
        object_detector.close()
        overlay_executor.shutdown(wait=False, cancel_futures=True)
        route_executor.shutdown(wait=False, cancel_futures=True)
        try:
            for pending in root.tk.call("after", "info"):
                root.tk.call("after", "cancel", pending)
        except tk.TclError:
            pass
        if camera:
            camera.close()
        if webcam:
            webcam.close()
        if passive:
            passive.close()
