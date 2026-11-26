"""Read raw motor positions and torque state without writing motor registers."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus


def snapshot(port):
    bus = SerialMotorsBus(
        port, {str(i): Motor(i, "sts3215", MotorNormMode.DEGREES) for i in range(1, 7)}
    )
    report = {
        "port": port,
        "read_only": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "positions_raw_by_id": {},
        "torque_enabled_by_id": {},
    }
    try:
        bus.connect(handshake=False)
        for i in range(1, 7):
            name = str(i)
            report["positions_raw_by_id"][name] = bus.read(
                "Present_Position", name, normalize=False, num_retry=2
            )
            report["torque_enabled_by_id"][name] = bus.read(
                "Torque_Enable", name, normalize=False, num_retry=2
            )
    finally:
        bus.disconnect(disable_torque=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--output", default="runs/joint_state.json")
    p.add_argument(
        "--reference", help="Previous snapshot, for signed raw position deltas"
    )
    args = p.parse_args()
    report = snapshot(args.port)
    if args.reference:
        before = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        if before["port"] != args.port:
            raise ValueError("Reference belongs to another serial port")
        report["delta_counts_by_id"] = {
            key: (value - before["positions_raw_by_id"][key] + 2048) % 4096 - 2048
            for key, value in report["positions_raw_by_id"].items()
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
