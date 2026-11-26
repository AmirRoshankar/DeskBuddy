"""Explicit operator-confirmed release/hold operations, never reaching motions."""

import argparse
import json
import time
from pathlib import Path

from .calibration_session import write_json
from .config import load_config
from .hold_position import hold_supported_pose


def supported_action(bus, config, action):
    if not bus.is_calibrated:
        raise ValueError("Motor calibration does not match hardware")
    if action == "hold":
        from .hardware_interlock import motion_disabled

        return hold_supported_pose(
            bus,
            config,
            stop_requested=lambda: (
                motion_disabled() or Path("runs/control/stop").exists()
            ),
        )
    if action != "release":
        raise ValueError("Unknown calibration action")
    bus.disable_torque(num_retry=2)
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    if len(torque) != 6 or any(torque.values()):
        raise RuntimeError("Not all motors report torque off. Keep the arm supported.")
    return dict(released=True, torque_enabled=torque, reach_executed=False)


def main():
    from .hardware_interlock import require_hardware_enabled

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("release", "hold"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "hold":
        require_hardware_enabled()
    config = load_config(args.config)
    pause = Path("runs/control/pause_robot")
    pause.parent.mkdir(parents=True, exist_ok=True)
    with pause.open("x") as stream:
        stream.write(f"calibration {args.action}")
    bus = None
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
                "Viewer did not release serial port; no action performed"
            )
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        bus = SO101Follower(
            SO101FollowerConfig(
                port=config["robot_port"],
                id=config["robot_id"],
                use_degrees=True,
                calibration_dir=Path(config["motor_calibration_dir"]),
            )
        ).bus
        bus.connect()
        result = supported_action(bus, config, args.action)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        pause.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
