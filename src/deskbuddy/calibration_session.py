"""Persist camera capture progress and automatically assess eight observations."""

import json
from pathlib import Path

from .live_snapshot import atomic_bytes
from .tool_calibration import fit_observations
from .tool_point_fit import check_tool_point, fit_tool_point


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def remember_session(session_id, config):
    write_json(
        "runs/calibration_active.json",
        {
            "session_id": session_id,
            "camera_serial": config["camera_serial"],
            "joint_offsets_deg": config["joint_offsets_deg"],
        },
    )


def restore_session(config):
    if config["calibration_verified"]:
        return None, []
    try:
        active = json.loads(Path("runs/calibration_active.json").read_text())
        if any(
            active.get(key) != config[key]
            for key in ("camera_serial", "joint_offsets_deg")
        ):
            return None, []
        rows = [
            json.loads(s)
            for s in Path("runs/tool_points.jsonl").read_text().splitlines()
            if s.strip()
        ]
        rows = [r for r in rows if r["session_id"] == active["session_id"]]
        return active["session_id"], rows
    except (OSError, ValueError, KeyError):
        return None, []


def assess_captures(rows):
    """Freeze the first eight observations; extra clicks do not alter this fit."""
    if len(rows) < 8:
        return None
    report = fit_observations(rows[:8])
    write_json(Path("runs/calibration") / rows[0]["session_id"] / "report.json", report)
    checks = ", ".join(f"{v:.1f}" for v in report["heldout_error_mm"])
    verdict = (
        "Fit passed; awaiting calibration review. Go remains disabled."
        if report["passed"]
        else "Fit needs review. Go remains disabled."
    )
    message = (
        f"8 points captured — automatic check: fit RMS {report['fit_rmse_mm']:.1f} mm, "
        f"max {report['fit_max_error_mm']:.1f} mm; check errors {checks} mm. {verdict}"
    )
    if not report["passed"]:
        candidate_path = (
            Path("runs/calibration")
            / rows[0]["session_id"]
            / "tool_point_candidate.json"
        )
        try:
            if candidate_path.exists():
                candidate = json.loads(candidate_path.read_text())
            else:
                candidate = fit_tool_point(rows[:8])
                write_json(candidate_path, candidate)
            if candidate["passed_initial_checks"]:
                if len(rows) < 10:
                    message += f" Jaw-tip offset candidate fits. Capture {10 - len(rows)} more varied poses to check it; no need to restart."
                else:
                    validation = check_tool_point(candidate, rows)
                    write_json(
                        candidate_path.with_name("tool_point_validation.json"),
                        validation,
                    )
                    checks = ", ".join(
                        f"{v:.1f}" for v in validation["fresh_check_error_mm"]
                    )
                    message = (
                        f"10 points captured — fresh jaw-tip checks: {checks} mm. "
                        + (
                            "Passed; awaiting review. "
                            if validation["passed"]
                            else "Needs review. "
                        )
                        + "Go remains disabled."
                    )
        except (ValueError, KeyError) as exc:
            message += f" Tool-point diagnostic: {exc}"
    return report, message


def invalidate_camera(config, config_path):
    # No motor commands: callers keep the existing torque state unchanged.
    # Keep the last *working* calibration: pressing Recalibrate twice must not
    # overwrite the backup with an already-invalidated configuration.
    if config.get("T_base_camera"):
        write_json("runs/calibration_before_reset.json", config)
    config["calibration_verified"] = False
    config["T_base_camera"] = None
    config["test_bed"] = None
    if config_path:
        write_json(config_path, config)


def checked_fit(rows, config):
    """Return only a same-camera fit that passes fresh recomputed checks."""
    if len(rows) < 8 or any(
        r["camera_serial"] != config["camera_serial"] for r in rows
    ):
        raise ValueError("Need eight observations from the configured camera")
    report = fit_observations(rows[:8])
    if report["passed"]:
        return {
            "T_base_camera": report["T_base_camera"],
            "tool_offset_m": config["tool_offset_m"],
        }
    path = (
        Path("runs/calibration") / rows[0]["session_id"] / "tool_point_candidate.json"
    )
    candidate = json.loads(path.read_text())
    validation = check_tool_point(candidate, rows)
    if not validation["passed"]:
        raise ValueError("Jaw-tip fit did not pass fresh checks")
    return {
        "T_base_camera": candidate["T_base_camera"],
        "tool_offset_m": candidate["tool_offset_m"],
    }
