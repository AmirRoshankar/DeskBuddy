"""Launch one explicit Go action with an immutable target/configuration snapshot."""

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnostics import tracking_fault
from .kinematics import JOINTS
from .live_snapshot import atomic_bytes
from .storage import save_plan


def read_process_log(path):
    """New workers use UTF-8; retain readable diagnostics from older Windows jobs."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


@dataclass
class MotionJob:
    process: object
    directory: Path
    stdout: object
    stderr: object

    def finish(self):
        self.stdout.close()
        self.stderr.close()
        report = self.directory / "result.json"
        return (
            json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
        )


def launch(plan, config, motor_state, popen=None):
    from .hardware_interlock import require_hardware_enabled

    require_hardware_enabled()
    if Path("runs/control/support_clearance_required").exists():
        raise ValueError(
            "Confirm arm holding and hands/support clear in the UI before Go"
        )
    if popen is None:
        popen = subprocess.Popen
    if not config["calibration_verified"] or not config["joint_mapping_verified"]:
        raise ValueError("Select a target after camera/model calibration")
    if motor_state is None or any(
        motor_state["torque_enabled"].get(j) != 1 for j in JOINTS
    ):
        raise ValueError("The five arm motors must already be holding before Go")
    fault = tracking_fault(motor_state)
    if fault:
        raise ValueError(fault)
    if Path("runs/control/pause_robot").exists():
        raise ValueError("Another operation currently owns the motor port")
    if not np.isfinite(plan.target_m).all():
        raise ValueError("Invalid selected target")
    folder = Path("runs/control/jobs") / str(uuid.uuid4())
    folder.mkdir(parents=True, exist_ok=False)
    save_plan(folder / "selected_plan.json", plan, config, synthetic=False)
    atomic_bytes(folder / "config.json", (json.dumps(config, indent=2) + "\n").encode())
    # A new deliberate Go action acknowledges a completed previous Stop request.
    Path("runs/control/stop").unlink(missing_ok=True)
    stdout = (folder / "stdout.log").open("w", encoding="utf-8")
    stderr = (folder / "stderr.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        "-m",
        "deskbuddy.execute_reach",
        "--config",
        str(folder / "config.json"),
        "--plan",
        str(folder / "selected_plan.json"),
        "--output",
        str(folder),
    ]
    try:
        process = popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            close_fds=True,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return MotionJob(process, folder, stdout, stderr)


def launch_calibration_action(action, config):
    """Caller must obtain explicit operator support confirmation before calling."""
    from .hardware_interlock import require_hardware_enabled

    if action == "hold":
        require_hardware_enabled()
    if action not in ("release", "hold"):
        raise ValueError("Unknown calibration action")
    if Path("runs/control/pause_robot").exists():
        raise ValueError("Another operation currently owns the motor port")
    if action == "hold":
        Path("runs/control/stop").unlink(missing_ok=True)
    folder = Path("runs/calibration-actions") / str(uuid.uuid4())
    folder.mkdir(parents=True)
    atomic_bytes(folder / "config.json", json.dumps(config).encode())
    stdout = (folder / "stdout.log").open("w", encoding="utf-8")
    stderr = (folder / "stderr.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                "-u",
                "-m",
                "deskbuddy.calibration_action",
                action,
                "--config",
                str(folder / "config.json"),
                "--output",
                str(folder / "result.json"),
            ],
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            close_fds=True,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return MotionJob(process, folder, stdout, stderr)
