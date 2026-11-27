"""Controlled ambiguity test: identical XYZ goals with two valid wrist postures."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..kinematics import Arm
from ..planning import check_workspace
from ..test_bed import table_clearance
from .data import check_budget, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--folder", default="runs/deskbuddy/multimodal")
    args = parser.parse_args()
    config = load_config(args.config)
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    check_budget(folder.parent, 1_000_000)
    start = np.deg2rad([0, -15, 30, 15, 3])
    rows = [[], [], []]
    for i, x in enumerate(np.linspace(0.205, 0.325, 17)):
        for j, y in enumerate(np.linspace(-0.08, 0.08, 17)):
            target = np.array([x, y, 0.10])
            solutions = []
            try:
                check_workspace(target, config)
                for wrist in (40.0, 90.0):
                    seed = start.copy()
                    seed[3] = np.deg2rad(wrist)
                    q = arm.ik(target, seed, locked_joints=["wrist_flex", "wrist_roll"])
                    if table_clearance(arm.frames(q)[0][2:], config).min() < 0.002:
                        raise ValueError("Endpoint too near table")
                    solutions.append(q)
            except ValueError:
                continue
            # Entire target pairs are held out together; no label leakage.
            bucket = (i * 13 + j * 7) % 10
            split = 1 if bucket == 8 else 2 if bucket == 9 else 0
            for q in solutions:
                rows[split].append((np.r_[start, target], q[:4] - start[:4]))
    if min(map(len, rows)) < 10:
        raise ValueError("Too few paired targets for this workspace")
    arrays = {}
    for name, records in zip(("train", "validation", "test"), rows):
        arrays[name + "_x"] = np.array([r[0] for r in records], dtype=np.float32)
        arrays[name + "_y"] = np.array([r[1] for r in records], dtype=np.float32)
    np.savez_compressed(folder / "dataset.npz", **arrays)
    metadata = dict(
        config=config,
        signature=signature(config),
        split_method="target-pair grouped spatial grid hash; balanced wrist 40/90 degree modes",
        splits=dict(zip(("train", "validation", "test"), map(len, rows))),
        motor_io=False,
        experiment="controlled multimodal stress test, not natural demonstration data",
    )
    (folder / "dataset.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["splits"]))


if __name__ == "__main__":
    main()
