"""Read-only serial timing benchmark. Never writes goals, gains, or torque."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .kinematics import JOINTS
from .live_snapshot import atomic_bytes


def summarize(values):
    a = np.asarray(values, float) * 1000
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }


def benchmark(
    bus,
    rates=(20, 40, 60, 80, 100),
    seconds=3.0,
    clock=time.perf_counter,
    sleep=time.sleep,
):
    names = list(JOINTS)

    def read(paired=False):
        values = bus.sync_read("Present_Position", names, num_retry=0)
        if set(values) != set(names) or not np.isfinite(list(values.values())).all():
            raise ValueError("Invalid position feedback")
        if paired:
            bus.sync_read("Goal_Position", names, num_retry=0)

    for _ in range(5):
        read()
    raw = []
    for _ in range(300):
        start = clock()
        read()
        raw.append(clock() - start)
    results = []
    for paired in (False, True):
        for hz in rates:
            period = 1 / hz
            durations = []
            lateness = []
            missed = 0
            began = clock()
            for index in range(max(1, math.ceil(seconds * hz))):
                scheduled = began + index * period
                wait = scheduled - clock()
                if wait > 0:
                    sleep(wait)
                start = clock()
                lateness.append(max(0.0, start - scheduled))
                read(paired)
                ended = clock()
                durations.append(ended - start)
                missed += int(ended > scheduled + period)
            row = {
                "mode": "two_reads_proxy" if paired else "position_read",
                "requested_hz": hz,
                "samples": len(durations),
                "period_ms": period * 1000,
                "latency": summarize(durations),
                "wake_lateness": summarize(lateness),
                "missed_deadlines": missed,
                "observed_hz": len(durations) / (clock() - began),
            }
            # Reserve half the cycle for command writes, calculations and jitter.
            row["candidate_with_headroom"] = bool(
                missed == 0 and row["latency"]["p99_ms"] < period * 1000 * 0.5
            )
            results.append(row)
    candidates = [
        r["requested_hz"]
        for r in results
        if r["mode"] == "two_reads_proxy" and r["candidate_with_headroom"]
    ]
    return {
        "raw_position_read": summarize(raw),
        "sweeps": results,
        "candidate_hz": max(candidates) if candidates else None,
        "position_commands_sent": False,
        "torque_commands_sent": False,
        "note": "Two reads are a conservative traffic proxy, not a measured read+write cycle. Candidate rates require full-loop timing and tracking validation during an explicit Go trial. Highest tested rate is not a proven hardware maximum.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--rates", nargs="+", type=float, default=[20, 40, 60, 80, 100])
    parser.add_argument("--output", default="runs/timing_benchmark.json")
    args = parser.parse_args()
    if not 0.1 <= args.seconds <= 10:
        parser.error("Use 0.1–10 seconds per rate")
    if any(not np.isfinite(rate) or not 1 <= rate <= 500 for rate in args.rates):
        parser.error("Benchmark rates must be finite and between 1 and 500 Hz")
    config = load_config(args.config)
    pause = Path("runs/control/pause_robot")
    pause.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pause.open("x") as stream:
            stream.write("read-only timing benchmark")
    except FileExistsError:
        parser.error("A motion or motor operation is active; wait for it to finish")
    bus = None
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            state = json.loads(Path("runs/live/state.json").read_text())
            if state.get("robot_io_paused") and time.time() - state["unix_time_s"] < 2:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Viewer did not release motor port")
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
        bus.connect()
        torque_before = bus.sync_read("Torque_Enable", normalize=False, num_retry=0)
        report = benchmark(bus, rates=args.rates, seconds=args.seconds)
        report["torque_before"] = torque_before
        report["torque_after"] = bus.sync_read(
            "Torque_Enable", normalize=False, num_retry=0
        )
        atomic_bytes(args.output, (json.dumps(report, indent=2) + "\n").encode())
        print(
            json.dumps({k: v for k, v in report.items() if k != "sweeps"}, indent=2),
            flush=True,
        )
    finally:
        if bus is not None and bus.is_connected:
            bus.disconnect(disable_torque=False)
        pause.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
