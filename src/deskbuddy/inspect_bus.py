"""Read-only motor identity check. No robot.connect(), configuration, or torque writes."""

import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus


def inspect(port, baudrate=1_000_000):
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    bus = SerialMotorsBus(
        port,
        {
            name: Motor(i, "sts3215", MotorNormMode.DEGREES)
            for i, name in enumerate(names, 1)
        },
    )
    # This changes only the host serial speed, not the motors' EEPROM baud rates.
    bus.default_baudrate = baudrate
    report = {"port": port, "baudrate": baudrate, "read_only": True, "responses": []}
    try:
        bus.connect(handshake=False)
        for i, name in enumerate(names, 1):
            models = [bus.ping(i, num_retry=1) for _ in range(2)]
            report["responses"].append(
                {"id": i, "expected_joint": name, "model_reads": models}
            )
    finally:
        # The default disconnect disables torque; explicitly avoid that write here.
        bus.disconnect(disable_torque=False)
    report["all_expected_ids_respond"] = all(
        row["model_reads"] == [777, 777] for row in report["responses"]
    )
    report["note"] = (
        "ID responses do not prove physical joint order or rule out duplicate IDs. "
        "No settings, positions, or torque registers were written."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--output", default="runs/motor_identity.json")
    args = parser.parse_args()
    report = inspect(args.port, args.baudrate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
