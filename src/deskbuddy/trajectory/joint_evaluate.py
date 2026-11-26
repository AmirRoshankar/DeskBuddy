"""Held-out full-arm validation of matched joint-space models, without hardware."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..kinematics import Arm
from ..table_collision import minimum_clearance
from .execution import check_pose_obstacles
from .joint_dataset import METHODS, condition, decode_path
from .joint_planner import check_self_collision
from .learning import Dense, Field, sample


def invalid_reason(arm, path, case):
    cfg = case["config"]
    if not np.isfinite(path).all():
        return "nonfinite"
    if (path < arm.lower - 1e-6).any() or (path > arm.upper + 1e-6).any():
        return "joint_limits"
    if (
        "wrist_roll" in cfg["locked_joints"]
        and np.max(abs(path[:, 4] - path[0, 4])) > 1e-6
    ):
        return "locked_joint"
    for a, b in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(abs(b - a)) / np.deg2rad(0.5))))
        if count > 2000:
            return "excessive_joint_step"
        for q in (a + (b - a) * t for t in np.linspace(0, 1, count + 1)):
            points = arm.frames(q)[0]
            if minimum_clearance(arm, q, cfg) < -1e-7:
                return "table"
            tip = points[-1]
            if (tip < np.array(cfg["workspace_min_m"]) - 1e-7).any() or (
                tip > np.array(cfg["workspace_max_m"]) + 1e-7
            ).any():
                return "workspace"
            try:
                check_self_collision(arm, q)
            except ValueError:
                return "self_collision"
            try:
                check_pose_obstacles(arm, q, case["boxes"])
            except ValueError:
                return "object_collision"
    return None


def load_models(folder, dataset_hash):
    models, checkpoints = {}, {}
    for name in METHODS:
        cp = torch.load(
            Path(folder) / f"{name}.pt", weights_only=True, map_location="cpu"
        )
        if cp["task"] != "joint_trajectory_3d" or cp["dataset_sha256"] != dataset_hash:
            raise ValueError("Models must share this exact full-arm dataset")
        model = (
            Dense(cp["condition_size"], cp["state_size"], cp["width"])
            if name == "mlp"
            else Field(
                cp["width"],
                cp.get("auxiliary_velocity_head", False),
                cp["condition_size"],
                cp["state_size"],
            )
        )
        model.load_state_dict(cp["state_dict"])
        models[name], checkpoints[name] = model.eval(), cp
    return models, checkpoints


def evaluate(dataset, models_folder, output, limit=100, candidates=4):
    if candidates not in (1, 4):
        raise ValueError("Use one or four candidates")
    source = Path(dataset)
    metadata = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((source / "dataset.npz").read_bytes()).hexdigest()
    models, checkpoints = load_models(models_folder, digest)
    torch.set_num_threads(1)
    limits = metadata["limits_rad"]
    arm = Arm(np.rad2deg(limits), metadata["tool_offset_m"])
    test = [
        (i, c)
        for i, c in enumerate(metadata["scenarios"])
        if c["split"] == "test"
        and c["goal"] is not None
        and c.get("endpoint_body_valid", True)
    ]
    # Spread a limited evaluation across families instead of taking one layout first.
    if len(test) > limit:
        test = [test[i] for i in np.linspace(0, len(test) - 1, limit, dtype=int)]
    records = []
    for index, case in test:
        noise = torch.randn(
            candidates, 150, generator=torch.Generator().manual_seed(1729 + index)
        )
        for name in ("direct", *METHODS):
            count = candidates if name in ("fm", "imf") else 1
            began = time.perf_counter()
            if name == "direct":
                paths = [np.linspace(case["start"], case["goal"], 32)]
            else:
                n = checkpoints[name]["normalization"]
                c = torch.tensor(
                    (condition(case, limits) - n["c_mean"]) / n["c_std"],
                    dtype=torch.float32,
                ).repeat(count, 1)
                with torch.inference_mode():
                    value = sample(
                        models[name], c, noise[:count], name, 8 if name == "fm" else 1
                    ).numpy()
                paths = [
                    decode_path(p, case, limits)
                    for p in value * n["y_std"] + n["y_mean"]
                ]
                if "wrist_roll" in case["config"]["locked_joints"]:
                    # Exact known boundary condition, also enforced by the runtime checker.
                    for p in paths:
                        p[:, 4] = case["start"][4]
            generation_ms = (time.perf_counter() - began) * 1000
            began = time.perf_counter()
            reasons = [invalid_reason(arm, p, case) for p in paths]
            valid = [p for p, reason in zip(paths, reasons) if reason is None]
            records.append(
                dict(
                    scenario=index,
                    family=case["family"],
                    method=name,
                    candidates=count,
                    valid=bool(valid),
                    raw_valid_candidates=len(valid),
                    reasons=reasons,
                    generation_ms=generation_ms,
                    validation_ms=(time.perf_counter() - began) * 1000,
                    direct_blocked=case["direct_blocked"],
                    obstacles=len(case["boxes"]),
                )
            )
        print(f"Evaluated {len(records) // 4}/{len(test)} scenes", flush=True)
    summary = {}
    for name in ("direct", *METHODS):
        rows = [r for r in records if r["method"] == name]
        blocked = [r for r in rows if r["direct_blocked"]]
        summary[name] = dict(
            scenes=len(rows),
            valid=sum(r["valid"] for r in rows),
            valid_rate=np.mean([r["valid"] for r in rows]).item() if rows else None,
            blocked_scenes=len(blocked),
            blocked_valid=sum(r["valid"] for r in blocked),
            median_generation_ms=float(np.median([r["generation_ms"] for r in rows])),
            failures={
                reason: sum(reason in r["reasons"] for r in rows)
                for reason in {x for r in rows for x in r["reasons"] if x}
            },
        )
    report = dict(
        task="joint_trajectory_3d",
        dataset_sha256=digest,
        motor_io=False,
        candidates=candidates,
        models=str(models_folder),
        summary=summary,
        records=records,
        notes=[
            "CPU inference; FM 8 evaluations, iMF 1; MLP deterministic one candidate.",
            "Raw means before repair/RRT fallback; exact endpoints and locked joints are imposed.",
            "Validity is geometric, not measured physical success or grasp success.",
            "Failure categories may overlap across candidates.",
        ],
    )
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/flowreach/joint-trajectories-v1")
    parser.add_argument(
        "--models", default="runs/flowreach/joint-trajectories-v1/models-v1"
    )
    parser.add_argument(
        "--output", default="runs/flowreach/joint-trajectories-v1/evaluation.json"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.dataset, args.models, args.output, args.limit, args.candidates)


if __name__ == "__main__":
    main()
