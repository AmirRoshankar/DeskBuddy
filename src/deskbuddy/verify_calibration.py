"""Read back saved calibration and positions; never configure or enable motors."""

import argparse
import json
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def verify(port, directory, robot_id):
    robot = SO101Follower(
        SO101FollowerConfig(
            port=port, id=robot_id, calibration_dir=Path(directory), use_degrees=True
        )
    )
    if not robot.calibration:
        raise ValueError("Saved calibration file is missing or empty")
    bus = robot.bus
    try:
        bus.connect()
        report = {
            "port": port,
            "read_only": True,
            "saved_calibration_matches_motors": bus.is_calibrated,
            "positions": bus.sync_read("Present_Position", num_retry=3),
            "torque_enabled": bus.sync_read(
                "Torque_Enable", normalize=False, num_retry=3
            ),
            "calibration_file": str(robot.calibration_fpath),
        }
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="deskbuddy")
    p.add_argument("--directory", default="runs/motor-calibration")
    p.add_argument("--output", default="runs/calibration_verification.json")
    args = p.parse_args()
    report = verify(args.port, args.directory, args.id)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["saved_calibration_matches_motors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
