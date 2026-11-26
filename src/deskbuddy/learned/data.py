"""Generate synthetic teacher IK pairs without camera, serial or motor imports."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from ..kinematics import Arm
from ..planning import check_workspace
from ..test_bed import table_clearance

MAX_BYTES = 10_000_000_000


def check_budget(folder, additional=0):
    folder = Path(folder).resolve()
    for candidate in (folder, *folder.parents):
        if (candidate / ".resource-budget.json").is_file():
            folder = candidate
            break
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if total + additional > MAX_BYTES:
        raise RuntimeError("FlowReach folder would exceed the 10 GB operator limit")
    return total


def signature(config):
    fields = {
        k: config[k]
        for k in (
            "model_joint_limits_deg",
            "tool_offset_m",
            "locked_joints",
            "joint_offsets_deg",
            "workspace_min_m",
            "workspace_max_m",
            "test_bed",
        )
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--folder", default="runs/flowreach")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    args = parser.parse_args()
    if (
        min(args.train, args.validation, args.test) < 1
        or args.train + args.validation + args.test > 500000
    ):
        parser.error("Use positive split sizes totaling at most 500,000")
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    check_budget(folder, (args.train + args.validation + args.test) * 256)
    config = load_config(args.config)
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    rng = np.random.default_rng(20261124)
    wanted = [args.train, args.validation, args.test]
    records = [[], [], []]
    begun = time.perf_counter()
    attempts = 0
    roll = np.deg2rad(config.get("validated_wrist_roll_deg") or 3.0)
    while any(len(r) < n for r, n in zip(records, wanted)):
        attempts += 1
        if attempts > sum(wanted) * 2000:
            raise RuntimeError(
                "Insufficient valid samples; no hardware action attempted"
            )
        goal = rng.uniform(arm.lower + 0.015, arm.upper - 0.015)
        goal[4] = roll
        target = arm.fk(goal)
        try:
            check_workspace(target, config)
        except ValueError:
            continue
        # Hold out complete 4 cm target cells, not alternate rows of one path.
        cell = np.floor(target / 0.04).astype(int)
        bucket = int((cell @ [73856093, 19349663, 83492791]) % 8)
        split = 1 if bucket == 6 else 2 if bucket == 7 else 0
        if len(records[split]) >= wanted[split]:
            continue
        start = np.clip(
            goal + rng.normal(0, np.deg2rad(25), 5),
            arm.lower + 0.015,
            arm.upper - 0.015,
        )
        start[4] = roll
        if (
            min(
                table_clearance(arm.frames(start)[0][2:], config).min(),
                table_clearance(arm.frames(goal)[0][2:], config).min(),
            )
            < 0.002
        ):
            continue
        try:
            label = arm.ik(
                target, start, config["ik_tolerance_m"], config["locked_joints"]
            )
        except ValueError:
            continue
        if table_clearance(arm.frames(label)[0][2:], config).min() < 0.002:
            continue
        records[split].append((np.r_[start, target], label[:4] - start[:4]))
        count = sum(map(len, records))
        if count % 250 == 0:
            print(
                json.dumps(
                    dict(
                        samples=count,
                        attempts=attempts,
                        elapsed_s=round(time.perf_counter() - begun, 1),
                    )
                ),
                flush=True,
            )
    arrays = {}
    for name, rows in zip(("train", "validation", "test"), records):
        arrays[name + "_x"] = np.array([r[0] for r in rows], dtype=np.float32)
        arrays[name + "_y"] = np.array([r[1] for r in rows], dtype=np.float32)
    np.savez_compressed(folder / "dataset.npz", **arrays)
    metadata = dict(
        config=config,
        signature=signature(config),
        seed=20261124,
        splits=dict(zip(("train", "validation", "test"), wanted)),
        split_method="disjoint 4 cm target cells; bucket hash mod 8",
        attempts=attempts,
        elapsed_s=time.perf_counter() - begun,
        physical_demonstrations=0,
        motor_io=False,
    )
    (folder / "dataset.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(done=True, bytes=check_budget(folder), elapsed_s=metadata["elapsed_s"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
