"""UI-confirmed hold enable; never generates or executes a reach."""

import time
from pathlib import Path
from tkinter import messagebox

from .control_plan import arm_holding
from .go_control import launch_calibration_action, read_process_log
from .hardware_interlock import disable_motion, enable_motion, motion_disabled


class HoldControl:
    def __init__(self, root, config, state, status, bar):
        self.root, self.config, self.state, self.status, self.bar = (
            root,
            config,
            state,
            status,
            bar,
        )
        self.clearance_marker = Path("runs/control/support_clearance_required")
        if self.clearance_marker.exists():
            state["hold_pending_clear"] = True
            self.bar.step(
                "Confirm hold and remove support before reaching.",
                "Holding · hands clear",
                self.finish,
            )

    def release(self):
        """Explicit Disable-button request: stop workers, then release all six motors."""
        disable_motion()
        self.state["release_pending"] = True
        self.state["plan"] = None
        self.status.set(
            "Disabling · support arm · waiting for motion to stop before torque release"
        )
        self.release_deadline = time.monotonic() + 15
        self.root.after(50, self.release_when_idle)

    def release_when_idle(self):
        state = self.state
        if (
            state.get("motion_job") is not None
            or state.get("calibration_job") is not None
            or Path("runs/control/pause_robot").exists()
        ):
            if time.monotonic() > self.release_deadline:
                state["release_pending"] = False
                self.status.set(
                    "Disable stopped motion; torque release could not acquire the port. Keep arm supported and retry."
                )
                return
            self.root.after(100, self.release_when_idle)
            return
        try:
            state["calibration_job"] = launch_calibration_action("release", self.config)
            self.root.after(100, self.poll_release)
        except (OSError, ValueError, RuntimeError) as exc:
            state["release_pending"] = False
            self.status.set(f"Torque release failed: {exc}")

    def poll_release(self):
        state = self.state
        job = state["calibration_job"]
        if job.process.poll() is None:
            self.root.after(100, self.poll_release)
            return
        try:
            report = job.finish()
            if job.process.returncode or not report or not report.get("released"):
                raise ValueError("Motors did not confirm torque off")
            state["hold_pending_clear"] = False
            state["hold_enabling"] = False
            self.clearance_marker.unlink(missing_ok=True)
            if state.get("motor") is not None:
                state["motor"]["torque_enabled"] = report["torque_enabled"]
            self.bar.grid_remove()
            self.status.set(
                "Disabled · all six motors released · move arm by hand; Enable locks its new pose"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.set(
                f"Disable could not verify release: {exc}. Keep arm supported."
            )
        finally:
            state["calibration_job"] = None
            state["release_pending"] = False

    def start(self, confirmed=False):
        state = self.state
        if (
            state.get("motor") is None
            or time.monotonic() - state["last_state_read"] > 0.5
        ):
            self.status.set("Wait for fresh motor feedback before enabling.")
            return
        if not confirmed and not messagebox.askyesno(
            "Enable motor hold",
            "Clamp the base and support the arm so it cannot fall. Is it stationary and securely supported? Enable holds the current pose. Leave support in place until checks finish.",
            parent=self.root,
        ):
            return
        try:
            enable_motion()
            self.clearance_marker.parent.mkdir(parents=True, exist_ok=True)
            self.clearance_marker.touch()
            state["calibration_job"] = launch_calibration_action("hold", self.config)
            state["hold_enabling"] = True
            state["hold_pending_clear"] = True
            state["plan"] = None
            self.bar.step(
                "Enabling hold at the measured pose. Keep the support in place."
            )
            self.root.after(100, self.poll)
        except (OSError, ValueError, RuntimeError) as exc:
            disable_motion()
            self.status.set(f"Enable failed: {exc}")

    def poll(self):
        state = self.state
        job = state["calibration_job"]
        if job.process.poll() is None:
            self.root.after(100, self.poll)
            return
        try:
            report = job.finish()
            if job.process.returncode or not report or not report.get("holding"):
                detail = (
                    read_process_log(job.directory / "stderr.log").strip().splitlines()
                )
                raise ValueError(detail[-1] if detail else "Hold verification failed")
            self.status.set(
                "Hold enabled. Remove support carefully, then confirm hands clear."
            )
            self.bar.step(
                "Hold verified with support. Remove the support carefully and clear your hands.",
                "Holding · hands clear",
                self.finish,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            disable_motion()
            self.status.set(f"Enable stopped: {exc}. Keep the arm supported.")
            self.bar.step(
                "Keep the arm supported. Enable retries if hold is off.",
                "Verify hold · hands clear",
                self.finish,
            )
        finally:
            state["calibration_job"] = None
            state["hold_enabling"] = False

    def finish(self):
        state = self.state
        if motion_disabled():
            self.status.set("Disabled. Press Enable before confirming hands clear.")
            return
        if (
            state.get("calibration_job") is not None
            or state.get("robot_io_paused")
            or time.monotonic() - state["last_state_read"] > 0.5
            or not arm_holding(state.get("motor"))
        ):
            self.status.set("Wait for fresh feedback confirming hold.")
            return
        state["hold_pending_clear"] = False
        self.clearance_marker.unlink(missing_ok=True)
        self.bar.grid_remove()
        self.status.set("Enabled. Select a target, then Go.")
