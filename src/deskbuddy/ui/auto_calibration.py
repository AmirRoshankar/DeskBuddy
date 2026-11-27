"""Operator-started automatic marker calibration, using checked joint moves."""

import json
import time
from copy import deepcopy
from pathlib import Path
from tkinter import messagebox

import numpy as np

from ..kinematics import JOINTS
from ..orange_calibration import (
    detect_orange_marker,
)


class AutoCalibration:
    def __init__(self, root, config, state, arm, status, launch, executor):
        self.root, self.config, self.state, self.arm = root, config, state, arm
        self.status, self.launch, self.executor = status, launch, executor
        self.samples, self.points, self.index = [], [], 0
        self.job = self.future = None
        self.stopped = False
        self.last_pixels = [None, None]
        self.last_webcam_image = None

    def stop(self):
        self.stopped = True

    def start(self):

        c, state = self.config, self.state
        if state.get("motion_job") or state.get("robot_io_paused"):
            raise ValueError("Finish the current operation first")
        second = c.get("secondary_camera_calibration") or {}
        if (
            not c.get("T_base_camera")
            or not second.get("T_base_camera")
            or not second.get("intrinsics")
        ):
            raise ValueError(
                "Save initial camera alignment first; an estimated pose is sufficient"
            )
        # Calibration is allowed to improve an unverified initial estimate.
        # Collect RGB observations directly instead of requiring triangulated
        # points from the camera poses we are trying to calibrate.
        self.bootstrap = not c.get("calibration_verified") or not second.get(
            "metric_3d_validated"
        )
        self.rgb_observations = []
        motor = state.get("motor") or {}
        if any((motor.get("torque_enabled") or {}).get(j) != 1 for j in JOINTS):
            raise ValueError(
                "Support the arm and enable position hold before automatic calibration"
            )
        if state.get("frame") is None or state.get("secondary_frame") is None:
            raise ValueError("Wait for both camera views")
        if not (state.get("tip_tracking") or {}).get("valid"):
            raise ValueError(
                "Click Select tip once in the webcam before automatic calibration; orange detection is not used to choose the webcam tip"
            )
        if not messagebox.askyesno(
            "Automatic calibration",
            "The arm will move through small checked joint poses. RGB camera poses will be fitted from these observations; initial camera accuracy is unverified. Keep the selected fixed jaw tip visible and hands clear. Start?",
        ):
            return
        start = np.array(state["q"], float)
        self.status.set(
            "Preflighting calibration poses and whole-arm routes; no movement yet"
        )
        self.begin_preflight(start)

    def begin_preflight(self, start):
        from ..calibration_poses import choose_poses

        self.future = self.executor.submit(
            choose_poses, self.arm, start, deepcopy(self.config), self.plan_next
        )
        self.root.after(100, self.poll_preflight)

    def poll_preflight(self):
        if self.stopped:
            return
        if not self.future.done():
            self.root.after(100, self.poll_preflight)
            return
        try:
            self.targets, summary = self.future.result()
            self.status.set(
                f"Preflight ready: {summary['accepted']} reachable poses, {summary['rejected']} blocked candidates skipped"
            )
            self.capture()
        except (ValueError, RuntimeError) as exc:
            self.status.set(f"Calibration preflight: {exc}. No movement started.")

    def capture(self):
        if self.stopped:
            self.status.set("Automatic calibration stopped; hold retained")
            return
        state = self.state
        if state.get("frame") is None or state.get("secondary_frame") is None:
            self.status.set("Automatic calibration stopped: camera frame unavailable")
            return
        if (
            time.time() - state["secondary_frame"][1] > 1
            or time.monotonic() - state.get("last_state_read", 0) > 0.5
        ):
            self.status.set(
                "Automatic calibration stopped: webcam or encoder feedback is stale"
            )
            return
        first, second = state["frame"][0], state["secondary_frame"][0]
        from ..orange_calibration import sheet_center_hints

        hints = sheet_center_hints(self.config, state["frame"][2])
        markers = [
            detect_orange_marker(frame, previous, initial_hint=hint)
            for frame, previous, hint in zip((first, second), self.last_pixels, hints)
        ]
        selected = state.get("tip_tracking")
        if selected is not None:
            markers[1] = (
                dict(pixel=selected["pixel"])
                if selected.get("valid")
                and time.time() - selected.get("unix_time_s", 0) < 1
                else None
            )
        elif self.last_webcam_image is not None and self.last_pixels[1] is not None:
            from ..orange_calibration import track_orange_marker

            markers[1] = track_orange_marker(
                self.last_webcam_image, second, self.last_pixels[1]
            )
        if markers[1] is None:
            self.status.set(
                "Webcam tip identity could not be maintained across poses; hold retained. RealSense is optional."
            )
            return
        self.last_webcam_image = second.copy()
        observed_pixels = [
            marker["pixel"] if marker is not None else [None, None]
            for marker in markers
        ]
        self.last_pixels = [
            marker["pixel"] if marker is not None else previous
            for marker, previous in zip(markers, self.last_pixels)
        ]
        intr = state["frame"][2]
        primary = dict(
            T_base_camera=self.config["T_base_camera"],
            intrinsics={
                k: intr[k] if isinstance(intr, dict) else getattr(intr, k)
                for k in ("fx", "fy", "ppx", "ppy")
            },
        )
        self.rgb_observations.append(deepcopy(observed_pixels))
        self.samples.append(np.array(state["q"], float))
        self.index += 1
        self.status.set(
            f"Automatic calibration: {self.index}/{len(self.targets)} · webcam"
            + (
                " + RealSense"
                if markers[0] is not None
                else " (RealSense tip unavailable; continuing)"
            )
        )
        if self.index == len(self.targets):
            try:
                from ..rgb_auto_calibration import fit_rgb_marker

                report = fit_rgb_marker(
                    self.arm,
                    self.samples,
                    self.rgb_observations,
                    [primary, self.config["secondary_camera_calibration"]],
                )
                report.update(
                    joint_samples_rad=[q.tolist() for q in self.samples],
                    marker_pixels=self.rgb_observations,
                )
                path = Path("runs/calibration/automatic-candidate.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                error = float(max(report["webcam_heldout_error_px"]))
                self.status.set(
                    f"Automatic calibration {'passed' if report['passed'] else 'failed'}: worst RGB check {error:.1f} px · candidate saved for review; current calibration unchanged"
                )
            except ValueError as exc:
                self.status.set(f"Automatic calibration fit failed: {exc}")
            return
        self.future = self.executor.submit(
            self.plan_next, np.array(state["q"], float), self.targets[self.index]
        )
        self.root.after(100, self.poll_plan)

    def plan_next(self, start, goal):
        from ..trajectory.joint_planner import plan_joint_route

        config = deepcopy(self.config)
        config.update(
            locked_joints=[],
            # Private joint-motion snapshot only; do not certify the saved camera calibration.
            calibration_verified=True,
            allow_object_collisions=False,
            max_joint_speed_deg_s=min(8, config["max_joint_speed_deg_s"]),
        )
        boxes = list(self.state.get("obstacle_boxes", []))
        return plan_joint_route(
            self.arm, start, self.arm.fk(goal), config, boxes, 1, 0, goal_pose=goal
        ), config

    def poll_plan(self):
        if self.stopped:
            return
        if not self.future.done():
            self.root.after(100, self.poll_plan)
            return
        try:
            plan, config = self.future.result()
            self.job = self.launch(plan, config, self.state["motor"])
            self.state["motion_job"] = self.job
            self.root.after(100, self.poll_move)
        except (ValueError, RuntimeError, OSError) as exc:
            self.status.set(f"Automatic calibration stopped: {exc}")

    def poll_move(self):
        if self.job.process.poll() is None:
            self.root.after(100, self.poll_move)
            return
        code = self.job.process.returncode
        try:
            result = self.job.finish()
            self.state["motion_job"] = None
            if self.stopped or code != 0 or not result or not result.get("completed"):
                self.status.set(
                    "Automatic calibration motion stopped; position hold retained"
                )
                return
            self.root.after(1200, self.capture)
        except (OSError, ValueError) as exc:
            self.state["motion_job"] = None
            self.status.set(f"Automatic calibration stopped: {exc}")
