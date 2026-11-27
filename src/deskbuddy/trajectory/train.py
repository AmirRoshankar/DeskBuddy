"""Train matched obstacle-conditioned trajectory baselines on GPU, never hardware."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..learned.data import check_budget
from .learning import Dense, Field, condition, encode_path, loss, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/deskbuddy/trajectories-v1")
    parser.add_argument("--output", default="runs/deskbuddy/trajectories-v1/models-v1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--diagonal-fraction", type=float, default=0.75)
    parser.add_argument("--adaptive-power", type=float, default=1.0)
    parser.add_argument("--imf-auxiliary-head", action="store_true")
    args = parser.parse_args()
    if (
        min(args.steps, args.width, args.batch) < 1
        or args.lr <= 0
        or not 0 <= args.diagonal_fraction <= 1
        or not 0 <= args.adaptive_power <= 1
    ):
        parser.error("Positive steps, width and batch required")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.pt")):
        parser.error("Choose a new output folder; checkpoints are preserved")
    check_budget(folder, 30_000_000)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 10_000_000_000 / total))
    source = Path(args.dataset)
    metadata = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    data = np.load(source / "dataset.npz")
    dataset_hash = hashlib.sha256((source / "dataset.npz").read_bytes()).hexdigest()
    if metadata.get("dataset_sha256", dataset_hash) != dataset_hash:
        raise ValueError("Dataset fingerprint mismatch")
    joint_task = metadata.get("task") == "joint_trajectory_3d"
    if joint_task:
        from .joint_dataset import condition as joint_condition
        from .joint_dataset import encode_path as joint_encode

        def context(case):
            return joint_condition(case, metadata["limits_rad"])

        def encode(path, case):
            return joint_encode(path, case, metadata["limits_rad"])
    else:
        context, encode = condition, encode_path
    contexts = np.array(
        [context(metadata["scenarios"][i]) for i in data["scenario_index"]]
    )
    targets = np.array(
        [
            encode(p, metadata["scenarios"][i])
            for p, i in zip(data["paths"], data["scenario_index"])
        ],
        np.float32,
    )
    train = data["split"] == "train"
    validation = data["split"] == "validation"
    if not train.any() or not validation.any():
        raise ValueError("Need nonempty training and validation splits")
    condition_size, state_size = contexts.shape[1], targets.shape[1]
    norm = dict(
        c_mean=contexts[train].mean(0).tolist(),
        c_std=np.maximum(contexts[train].std(0), 0.05).tolist(),
        y_mean=targets[train].mean(0).tolist(),
        y_std=np.maximum(targets[train].std(0), 0.05).tolist(),
    )
    cx = torch.tensor(
        (contexts - norm["c_mean"]) / norm["c_std"], device=device, dtype=torch.float32
    )
    cy = torch.tensor(
        (targets - norm["y_mean"]) / norm["y_std"], device=device, dtype=torch.float32
    )
    tx, ty = cx[train], cy[train]
    vx, vy = cx[validation], cy[validation]
    models = {
        name: (
            Dense(condition_size, outputs=state_size, width=args.width)
            if name == "mlp"
            else Field(
                args.width,
                auxiliary=(name == "imf" and args.imf_auxiliary_head),
                condition_dim=condition_size,
                state_dim=state_size,
            )
        ).to(device)
        for name in ("mlp", "fm", "imf")
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        for name, model in models.items()
    }
    best = {name: float("inf") for name in models}
    val_noise = torch.randn((2, len(vx), state_size), device=device)
    started = time.perf_counter()
    print(
        json.dumps(
            dict(
                device=device,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                train=len(tx),
                validation=len(vx),
                parameters={
                    name: sum(p.numel() for p in m.parameters())
                    for name, m in models.items()
                },
            )
        ),
        flush=True,
    )
    with (folder / "training.jsonl").open("w", encoding="utf-8") as history:
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(tx), (args.batch,), device=device)
            x, y = tx[indices], ty[indices]
            losses = {}
            for name, model in models.items():
                optimizer = optimizers[name]
                optimizer.param_groups[0]["lr"] = args.lr * (
                    0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
                )
                optimizer.zero_grad(set_to_none=True)
                value = loss(
                    model,
                    x,
                    y,
                    name,
                    warmup=step <= min(1000, args.steps // 10),
                    diagonal_fraction=args.diagonal_fraction,
                    adaptive_power=args.adaptive_power,
                )
                if not torch.isfinite(value):
                    raise RuntimeError(f"Nonfinite {name} loss at {step}")
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses[name] = float(value.detach())
            if step % 500 == 0 or step == args.steps:
                report = dict(
                    step=step,
                    elapsed_s=time.perf_counter() - started,
                    train_loss=losses,
                    validation_energy_score={},
                )
                with torch.no_grad():
                    for name, model in models.items():
                        # Two fixed independent draws estimate the proper energy score.
                        # MLP is its degenerate distribution; FM validation uses 8 NFEs.
                        draws = [
                            sample(model, vx, n, name, 8 if name == "fm" else 1)
                            for n in val_noise
                        ]
                        score = float(
                            (
                                (draws[0] - vy).norm(dim=1)
                                + (draws[1] - vy).norm(dim=1)
                                - (draws[0] - draws[1]).norm(dim=1)
                            ).mean()
                            / 2
                        )
                        report["validation_energy_score"][name] = score
                        if score < best[name]:
                            best[name] = score
                            check_budget(folder, 5_000_000)
                            torch.save(
                                dict(
                                    schema=1,
                                    task="joint_trajectory_3d"
                                    if joint_task
                                    else "cartesian_trajectory_2d",
                                    method=name,
                                    width=args.width,
                                    state_dict={
                                        k: v.detach().cpu()
                                        for k, v in model.state_dict().items()
                                    },
                                    normalization=norm,
                                    step=step,
                                    validation_energy_score=score,
                                    seed=args.seed,
                                    dataset=str(source),
                                    dataset_sha256=dataset_hash,
                                    shared_batch_seed=args.seed,
                                    limits_rad=metadata.get("limits_rad"),
                                    tool_offset_m=metadata.get("tool_offset_m"),
                                    waypoints=32,
                                    condition_size=condition_size,
                                    state_size=state_size,
                                    auxiliary_velocity_head=(
                                        name == "imf" and args.imf_auxiliary_head
                                    ),
                                ),
                                folder / (name + ".pt"),
                            )
                report["peak_gpu_bytes"] = (
                    torch.cuda.max_memory_allocated() if device == "cuda" else 0
                )
                print(json.dumps(report), flush=True)
                history.write(json.dumps(report) + "\n")
                history.flush()
    summary = dict(
        device=device,
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        batch=args.batch,
        elapsed_s=time.perf_counter() - started,
        best_validation_energy_score=best,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        artifact_bytes=check_budget(folder),
        motor_io=False,
        dataset_sha256=dataset_hash,
        task=metadata.get("task", "cartesian_trajectory_2d"),
        methods=list(models),
        matched_training="same artifact, split, normalization, minibatch row IDs, update count, width, batch and LR schedule; method objectives and input/output sizes differ",
        learning_rate=args.lr,
        diagonal_fraction=args.diagonal_fraction,
        adaptive_power=args.adaptive_power,
        imf_auxiliary_head=args.imf_auxiliary_head,
        warmup_steps=min(1000, args.steps // 10),
        objective="MLP MSE; FM instantaneous-velocity MSE; iMF predicted diagonal-velocity JVP v-loss with adaptive weighted L2; no CFG.",
    )
    (folder / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
