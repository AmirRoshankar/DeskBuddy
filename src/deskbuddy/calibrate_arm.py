"""Chat-guided SDK calibration with no torque-enable or motion commands.

Connect the bus directly to avoid the robot connection routine's configuration stage.
Only the SDK's range-recorder keyboard handling is replaced, for piped/remote stdin.
"""

import argparse
import json
import threading
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def record_ranges(bus, motors=None, display_values=True):
    names = bus._get_motors_list(motors)
    start = bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
    mins, maxes = start.copy(), start.copy()
    finished = threading.Event()
    abort = threading.Event()

    def await_finish():
        try:
            input()
        except EOFError:
            abort.set()
        finally:
            finished.set()

    threading.Thread(target=await_finish, daemon=True).start()
    last_print = 0.0
    while not finished.is_set():
        positions = bus.sync_read(
            "Present_Position", names, normalize=False, num_retry=5
        )
        mins = {name: min(mins[name], positions[name]) for name in names}
        maxes = {name: max(maxes[name], positions[name]) for name in names}
        if display_values and time.monotonic() - last_print > 5:
            print(
                "RANGES "
                + json.dumps({name: [mins[name], maxes[name]] for name in names}),
                flush=True,
            )
            last_print = time.monotonic()
        time.sleep(0.05)
    if abort.is_set():
        raise RuntimeError("Input closed; calibration aborted without saving ranges")
    unmoved = [name for name in names if maxes[name] - mins[name] < 100]
    if unmoved:
        raise ValueError(
            f"Insufficient recorded travel for {unmoved}; ranges were not saved"
        )
    return mins, maxes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--id", default="deskbuddy")
    p.add_argument("--directory", default="runs/motor-calibration")
    args = p.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port, id=args.id, calibration_dir=directory, use_degrees=True
        )
    )
    bus = robot.bus
    try:
        bus.connect()
        backup = {}
        for name in bus.motors:
            backup[name] = {
                register: bus.read(register, name, normalize=False, num_retry=2)
                for register in (
                    "Homing_Offset",
                    "Min_Position_Limit",
                    "Max_Position_Limit",
                    "Torque_Enable",
                )
            }
        backup_path = directory / "pre_calibration_registers.json"
        if backup_path.exists():
            backup_path = directory / f"pre_calibration_registers_{time.time_ns()}.json"
        backup_path.write_text(json.dumps(backup, indent=2) + "\n", encoding="utf-8")
        print(f"Backed up existing calibration registers to {backup_path}", flush=True)
        # Keep the SDK's homing, range handling, EEPROM writes and file format.
        bus.record_ranges_of_motion = lambda motors=None, display_values=True: (
            record_ranges(bus, motors, display_values)
        )
        robot.calibrate()
        print("CALIBRATION_SAVED", str(robot.calibration_fpath), flush=True)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
