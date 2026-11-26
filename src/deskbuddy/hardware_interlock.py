"""Persistent operator-requested hardware pause; offline work remains available."""

from pathlib import Path


def motion_disabled():
    return Path("runs/control/hardware_disabled").exists()


def disable_motion():
    control = Path("runs/control")
    control.mkdir(parents=True, exist_ok=True)
    (control / "hardware_disabled").write_text("Disabled from UI", encoding="utf-8")
    (control / "stop").touch()


def enable_motion():
    # Enabling never launches a reach, releases torque, or clears a Stop.
    Path("runs/control/hardware_disabled").unlink(missing_ok=True)


def require_hardware_enabled():
    if motion_disabled():
        raise RuntimeError(
            "Hardware disabled by operator. Explicit resumption is required."
        )
