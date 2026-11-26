"""Matched held-out gravity/contact evaluation of geometric and physics-trained iMF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..trajectory.learning import Field, sample
from .can_physics import rollout
from .grasp_lab import unit


def evaluate(dataset, checkpoints, output):
    source = np.load(dataset)
    groups = source["object_id"]
    # One direction per independent held-out can; first 40 objects were the
    # separate proposal-tuning pilot and are excluded from all model claims.
    indices = []
    for group in np.unique(groups[(groups % 10 == 0) & (groups >= 40)]):
        indices.append(int(np.flatnonzero(groups == group)[0]))
    context = torch.tensor(source["contexts"][indices])
    parameters = source["physics_parameters"][indices]
    noise = torch.randn(len(context), 8, generator=torch.Generator().manual_seed(1739))
    torch.set_num_threads(1)
    report = dict(
        objects=len(indices),
        pilot_objects_excluded=40,
        matched_inputs_and_noise=True,
        gravity_m_s2=9.81,
        full_robot_simulated=False,
        motor_io=False,
        models={},
    )
    for name, path in checkpoints.items():
        cp = torch.load(path, map_location="cpu", weights_only=True)
        model = Field(cp["width"], condition_dim=5, state_dim=8)
        model.load_state_dict(cp["state_dict"])
        model.eval()
        with torch.inference_mode():
            predictions = sample(model, context, noise, "imf", 1)
            predictions[:, 3:7] = unit(predictions[:, 3:7])
        outcomes = []
        for c, p, params in zip(context.numpy(), predictions.numpy(), parameters):
            if not np.isfinite(p).all() or not 0.002 < float(p[7] * 0.05) < 0.025:
                outcomes.append(dict(success=False, pickup=False, invalid_output=True))
                continue
            outcomes.append(
                rollout(
                    float(c[0] * 0.05),
                    float(c[2] * 0.05),
                    p[:3] * 0.05,
                    p[3:7],
                    float(p[7] * 0.05),
                    mass=float(params[0]),
                    friction=float(params[1]),
                    force=float(params[2]),
                )
            )
        report["models"][name] = dict(
            checkpoint=str(path),
            nfe=1,
            physics_training=cp.get("physics_training", False),
            success_count=sum(row["success"] for row in outcomes),
            pick_transfer_drop_rate=float(
                np.mean([row["success"] for row in outcomes])
            ),
            pickup_rate=float(np.mean([row["pickup"] for row in outcomes])),
            physical_robot_success_measured=False,
            outcomes=outcomes,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        name: {k: v for k, v in values.items() if k != "outcomes"}
        for name, values in report["models"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--physics-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.dataset,
                dict(
                    geometric="runs/flowreach/can-rim-grasps-v1/imf.pt",
                    physics=args.physics_checkpoint,
                ),
                args.output,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
