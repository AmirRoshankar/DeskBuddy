"""Operator-driven calibration workflow, separate from camera rendering and reaching."""

import time
from tkinter import messagebox

from .calibration_session import checked_fit, write_json
from .go_control import launch_calibration_action, read_process_log
from .paper_calibration import apply_paper, fit_paper
from .ui.calibration import CalibrationBar


class CalibrationFlow:
    def __init__(
        self, root, config, state, status, config_path, on_fit, on_finish, on_restart
    ):
        self.root, self.config, self.state = root, config, state
        self.status, self.path = status, config_path
        self.on_fit, self.on_finish, self.on_restart = on_fit, on_finish, on_restart
        self.bar = CalibrationBar(root)
        self.stage = None
        self.pixels = []
        self.paper = None

    def save(self):
        if self.path:
            write_json(self.path, self.config)

    def begin(self):
        self.stage = "release"
        self.pixels, self.paper = [], None
        self.bar.step(
            "1 / 4 · Clamp the base and support the arm so it cannot fall. Release hold to reposition it by hand. No reach will run.",
            "Arm supported — release hold",
            lambda: self.support_action("release"),
        )

    def capture_step(self):
        self.stage = "capture"
        count = len(self.state["tool_points"])
        self.bar.step(
            f"2 / 4 · Arm points {count}/8. With torque off, close the jaws. Hand-guide to varied positions and heights, hold still 1 s, then click the same jaw-tip meeting point."
        )

    def support_action(self, action):
        if (
            self.state.get("calibration_job") is not None
            or self.state["motion_job"] is not None
        ):
            return
        prompt = (
            "Is the arm securely supported so it cannot fall? This turns off all six motors for hand-guided calibration."
            if action == "release"
            else "Is the arm supported and stationary in a comfortable pose above the paper? This enables hold at its current pose. Leave the support in place."
        )
        if not messagebox.askyesno("Confirm arm support", prompt, parent=self.root):
            return
        try:
            self.state["calibration_job"] = launch_calibration_action(
                action, self.config
            )
            self.bar.step("Checking motor state… Keep the arm supported.")
            self.root.after(100, self.poll, action)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.set(f"Calibration action blocked: {exc}")

    def poll(self, action):
        job = self.state["calibration_job"]
        if job.process.poll() is None:
            self.root.after(100, self.poll, action)
            return
        try:
            report = job.finish()
            if job.process.returncode or not report:
                lines = (
                    read_process_log(job.directory / "stderr.log").strip().splitlines()
                )
                raise ValueError(lines[-1] if lines else "Motor operation failed")
            if action == "release":
                self.capture_step()
                self.status.set(
                    "All six motors report torque off. Capture tool poses in the color view."
                )
            else:
                self.stage = "clear"
                self.bar.step(
                    "4 / 4 · Hold restored. Keep a hand nearby as you remove the support. Finish when the arm holds and box/hands are clear.",
                    "Arm holding, hands clear — finish",
                    self.finish,
                )
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.set(
                f"Calibration motor check failed: {exc}. Keep the arm supported."
            )
            self.bar.step(
                f"Motor check failed: {exc}. Keep the arm supported.",
                "Retry supported check",
                lambda: self.support_action(action),
            )
        finally:
            self.state["calibration_job"] = None

    def assessment(self, message):
        try:
            checked_fit(self.state["tool_points"], self.config)
            self.bar.step("2 / 4 · " + message, "Use checked camera fit", self.use_fit)
        except (OSError, ValueError, KeyError):
            self.bar.step(message, "Restart arm points", self.on_restart)

    def use_fit(self):
        try:
            self.config.update(checked_fit(self.state["tool_points"], self.config))
            self.config["calibration_verified"] = False
            self.on_fit()
            self.save()
            self.begin_paper()
        except (OSError, ValueError, KeyError) as exc:
            self.status.set(f"Fit cannot be applied: {exc}")

    def begin_paper(self):
        self.stage, self.pixels, self.paper = "paper", [], None
        self.bar.step(
            "3 / 4 · Flatten the A4 sheet. Click any corner, its neighbor along the SHORT (210 mm) edge, then the remaining corners around the perimeter. 0/4.",
            "Restart corners",
            self.begin_paper,
        )

    def capture_paper(self, pixel):
        if self.state.get("camera_stale"):
            raise ValueError("Wait for a fresh camera image before paper calibration")
        if len(self.pixels) >= 4:
            return
        self.pixels.append(list(pixel))
        if len(self.pixels) < 4:
            self.bar.step(
                f"Paper corners {len(self.pixels)}/4 · Continue around the perimeter. First edge must be the short 210 mm edge.",
                "Restart corners",
                self.begin_paper,
            )
            return
        try:
            _, depth, intrinsics = self.state["frame"]
            self.paper = fit_paper(
                self.pixels,
                intrinsics,
                self.config["T_base_camera"],
                depth,
                table_z=self.config["table_z_m"]
                if self.config.get("assume_level_table")
                else None,
            )
            error = max(self.paper["reprojection_error_px"])
            self.bar.step(
                f"Paper fit {error:.1f} px · Check the outline follows all four edges. Use the bounds or click Retry to redo the corners.",
                "Use paper bounds",
                self.use_paper,
                retry=self.begin_paper,
            )
        except (ValueError, RuntimeError) as exc:
            self.bar.step(str(exc), "Retry paper corners", self.begin_paper)
            raise

    def use_paper(self):
        try:
            self.config.update(apply_paper(self.config, self.paper))
            self.save()
            self.stage = "hold"
            self.bar.step(
                "4 / 4 · Paper bounds saved. Put the jaw tip above the sheet in a comfortable middle pose, support the arm and keep it still.",
                "Arm supported — restore hold",
                lambda: self.support_action("hold"),
            )
        except (OSError, ValueError, TypeError) as exc:
            self.status.set(f"Paper bounds rejected: {exc}")

    def finish(self):
        from .kinematics import JOINTS

        motor = self.state["motor"]
        if (
            motor is None
            or time.monotonic() - self.state["last_state_read"] > 0.5
            or not all(motor["torque_enabled"].get(j) == 1 for j in JOINTS)
        ):
            self.status.set(
                "Wait for fresh feedback confirming all five arm joints are holding."
            )
            return
        self.config["calibration_verified"] = True
        try:
            self.save()
        except OSError as exc:
            self.config["calibration_verified"] = False
            self.status.set(f"Calibration could not be saved: {exc}")
            return
        self.stage = None
        self.bar.grid_remove()
        self.on_finish()
        self.status.set(
            "Camera and A4 bounds saved. Click a point to preview; only Go starts a reach."
        )
