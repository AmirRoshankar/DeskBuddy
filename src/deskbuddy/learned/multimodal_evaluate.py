"""Evaluate distributions on paired-posture held-out goals; no test-time fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..kinematics import Arm
from .data import check_budget
from .inference import Proposal
from .train import energy_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/deskbuddy/multimodal")
    args = parser.parse_args()
    folder = Path(args.folder)
    meta = json.loads((folder / "dataset.json").read_text())
    config = meta["config"]
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    data = np.load(folder / "dataset.npz")
    inputs = data["test_x"].astype(float)
    outputs = data["test_y"].astype(float)
    models = {
        name: Proposal(folder / (name + ".pt"), config, name) for name in ("mlp", "imf")
    }
    unique = np.unique(inputs, axis=0)
    rows = []
    example = None
    for x in unique:
        start, target = x[:5], x[5:]
        labels = outputs[(inputs == x).all(1)]
        row = {}
        for name, wrapper in models.items():
            norm = wrapper.normalization
            condition = torch.tensor(
                (x - norm["x_mean"]) / norm["x_std"], dtype=torch.float32
            )[None]
            with torch.inference_mode():
                if name == "imf":
                    noise = torch.randn(
                        (64, 4), generator=torch.Generator().manual_seed(20261125)
                    )
                    generated = wrapper.model.sample(condition.repeat(64, 1), noise)
                else:
                    generated = wrapper.model(condition)
            delta = generated.numpy() * norm["y_std"] + norm["y_mean"]
            poses = np.tile(start, (len(delta), 1))
            poses[:, :4] += delta
            gaps = np.array([np.linalg.norm(arm.fk(q) - target) * 1000 for q in poses])
            bounded = ((poses >= arm.lower) & (poses <= arm.upper)).all(1)
            angles = np.rad2deg(poses[:, 3])
            modes = [
                bool((bounded & (gaps < 10) & (abs(angles - angle) < 5)).any())
                for angle in (40, 90)
            ]
            observed = torch.tensor(
                (labels - norm["y_mean"]) / norm["y_std"], dtype=torch.float32
            )
            score = float(
                energy_score(generated[None].repeat(len(labels), 1, 1), observed)
            )
            row[name] = dict(
                gaps_mm=gaps.tolist(),
                both_modes_covered=all(modes),
                success_2mm=float((bounded & (gaps < 2)).mean()),
                success_10mm=float((bounded & (gaps < 10)).mean()),
                probabilistic_energy_score=score,
            )
            if example is None:
                row[name]["poses_for_example"] = poses.tolist()
        if example is None:
            teachers = np.tile(start, (len(labels), 1))
            teachers[:, :4] += labels
            example = dict(
                start=start.tolist(),
                target=target.tolist(),
                teacher_poses=teachers.tolist(),
                mlp_poses=row["mlp"].pop("poses_for_example"),
                imf_poses=row["imf"].pop("poses_for_example"),
            )
        rows.append(row)
    summary = {}
    for name in models:
        summary[name] = dict(
            median_raw_model_gap_mm=float(
                np.median([g for row in rows for g in row[name]["gaps_mm"]])
            ),
            raw_success_2mm=float(np.mean([row[name]["success_2mm"] for row in rows])),
            raw_success_10mm=float(
                np.mean([row[name]["success_10mm"] for row in rows])
            ),
            both_modes_covered_fraction=float(
                np.mean([row[name]["both_modes_covered"] for row in rows])
            ),
            probabilistic_energy_score=float(
                np.mean([row[name]["probabilistic_energy_score"] for row in rows])
            ),
        )
    report = dict(
        controlled_stress_test=True,
        physical_tested=False,
        heldout_targets=len(unique),
        imf_samples_per_target=64,
        mode_definition="wrist flex within 5 degrees of 40 or 90, FK gap below 10 mm, within joint limits",
        probabilistic_score_is_not_motor_energy=True,
        summary=summary,
        example=example,
    )
    payload = json.dumps(report, indent=2)
    check_budget(folder, len(payload.encode()))
    (folder / "evaluation.json").write_text(payload, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
