"""Hold the measured supported pose. Never loads or executes a reaching plan."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .kinematics import JOINTS


def hold_supported_pose(bus, config, sleep=time.sleep, stop_requested=lambda: False):
    def check_stop():
        if stop_requested():
            raise RuntimeError("Hold enable stopped; keep the arm supported")

    check_stop()
    names = list(JOINTS)
    # This commissioning operation assumes the operator has left the support in place.
    if not bus.is_calibrated:
        raise ValueError("Saved motor calibration does not match hardware")
    modes = bus.sync_read("Operating_Mode", names, normalize=False, num_retry=2)
    torque_before = bus.sync_read("Torque_Enable", names, normalize=False, num_retry=2)
    if any(value != 0 for value in modes.values()):
        raise ValueError("All arm motors must already be in position mode")
    if any(value != 0 for value in torque_before.values()):
        raise ValueError(
            "This initial-hold operation expects all arm motor torque disabled"
        )
    first = bus.sync_read("Present_Position", names, normalize=False, num_retry=2)
    sleep(0.15)
    raw = bus.sync_read("Present_Position", names, normalize=False, num_retry=2)
    for sample in (first, raw):
        if set(sample) != set(names):
            raise ValueError("Incomplete encoder feedback; torque remains disabled")
        values = np.array(list(sample.values()), float)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 4095).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError("Invalid raw encoder feedback; torque remains disabled")
    if max(abs(raw[j] - first[j]) for j in names) > 8:
        raise ValueError(
            "Arm is moving; leave it supported and still before enabling hold"
        )
    # Holding a supported stationary pose does not request a planned trajectory.
    # Use verified raw encoder goals; model planning bounds remain enforced for reaches.
    # Acknowledged writes, then read-back, before ANY torque-enable command.
    for name in names:
        check_stop()
        bus.write("Goal_Position", name, int(raw[name]), normalize=False, num_retry=2)
    goals = bus.sync_read("Goal_Position", names, normalize=False, num_retry=2)
    if goals != raw:
        raise ValueError(
            "Hold goals did not read back exactly; torque remains disabled"
        )
    before_enable = bus.sync_read(
        "Present_Position", names, normalize=False, num_retry=2
    )
    if max(abs(before_enable[j] - raw[j]) for j in names) > 8:
        raise ValueError(
            "The supported pose changed while preparing hold; torque remains disabled"
        )
    check_stop()
    bus.enable_torque(names, num_retry=2)
    maximum_drift = {j: 0 for j in names}
    for _ in range(30):
        check_stop()
        measured = bus.sync_read(
            "Present_Position", names, normalize=False, num_retry=2
        )
        for name in names:
            maximum_drift[name] = max(
                maximum_drift[name], abs(measured[name] - raw[name])
            )
        if max(maximum_drift.values()) > 34:
            raise RuntimeError(
                "Hold moved more than 3 degrees; keep the support in place and inspect the arm"
            )
        sleep(0.1)
    torque_after = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    if any(torque_after[j] != 1 for j in names):
        raise RuntimeError(
            "Not all arm motors report torque enabled; keep the support in place"
        )
    return {
        "holding": True,
        "held_raw_positions": raw,
        "torque_enabled": torque_after,
        "max_drift_deg": {j: value * 360 / 4095 for j, value in maximum_drift.items()},
        "reach_executed": False,
        "gripper_unchanged": True,
        "note": "Verified while the support remained in place. Recheck after the operator removes it.",
    }


def main():
    from .hardware_interlock import require_hardware_enabled

    require_hardware_enabled()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--output", default="runs/hold_position.json")
    args = p.parse_args()
    config = load_config(args.config)
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
    try:
        bus.connect()
        from .hardware_interlock import motion_disabled

        result = hold_supported_pose(bus, config, stop_requested=motion_disabled)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        # Keep a successful hold active when releasing the serial port. Never auto-release here.
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
