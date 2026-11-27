"""Offline GPU training: supervised MLP baseline and improved MeanFlow IK proposal."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import check_budget
from .models import MeanFlow, Network, imf_loss


def energy_score(samples, observed):
    """Distribution score: E||X-y|| - 0.5 E||X-X'||; off-diagonal sample pairs."""
    count = samples.shape[1]
    accuracy = torch.linalg.vector_norm(samples - observed[:, None], dim=-1).mean(1)
    if count < 2:
        return accuracy.mean()
    diversity = torch.cdist(samples, samples).sum((1, 2)) / (count * (count - 1))
    return (accuracy - 0.5 * diversity).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/deskbuddy")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--validation-score", choices=("mse", "energy"), default="mse")
    args = parser.parse_args()
    if (
        not 1 <= args.steps <= 100000
        or not 1 <= args.batch <= 2048
        or not 16 <= args.width <= 512
    ):
        parser.error("Training size outside bounded experiment limits")
    folder = Path(args.folder)
    check_budget(folder, 20_000_000)
    torch.manual_seed(724)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.9, 10_000_000_000 / total))
    source = np.load(folder / "dataset.npz")
    x, y = source["train_x"], source["train_y"]
    normalization = dict(
        x_mean=x.mean(0).tolist(),
        x_std=np.maximum(x.std(0), 0.01).tolist(),
        y_mean=y.mean(0).tolist(),
        y_std=np.maximum(y.std(0), 0.01).tolist(),
        x_min=x.min(0).tolist(),
        x_max=x.max(0).tolist(),
    )
    tensors = {}
    for split in ("train", "validation"):
        tensors[split] = (
            torch.tensor(
                (source[split + "_x"] - normalization["x_mean"])
                / normalization["x_std"],
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                (source[split + "_y"] - normalization["y_mean"])
                / normalization["y_std"],
                device=device,
                dtype=torch.float32,
            ),
        )
    models = dict(
        mlp=Network(8, args.width).to(device), imf=MeanFlow(args.width).to(device)
    )
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
        for name, model in models.items()
    }
    best = dict.fromkeys(models, float("inf"))
    metadata = json.loads((folder / "dataset.json").read_text())
    begun = time.perf_counter()
    history = (folder / "training.jsonl").open("w", encoding="utf-8")
    val_x, val_y = tensors["validation"]
    val_noise = torch.randn_like(val_y)
    score_noise = (
        torch.randn((len(val_y), 16, 4), device=device)
        if args.validation_score == "energy"
        else None
    )
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(x), (args.batch,), device=device)
        cx, cy = (t[indices] for t in tensors["train"])
        losses = {}
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.param_groups[0]["lr"] = (
                0.00005 + 0.00095 * (1 + math.cos(math.pi * step / args.steps)) / 2
            )
            optimizer.zero_grad(set_to_none=True)
            loss = (
                (model(cx) - cy).square().mean()
                if name == "mlp"
                else imf_loss(model, cx, cy, warmup=step <= min(1000, args.steps // 10))
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} training loss at {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses[name] = float(loss.detach())
        if step % 500 == 0 or step == args.steps:
            report = dict(
                step=step,
                elapsed_s=round(time.perf_counter() - begun, 1),
                train=losses,
                validation={},
            )
            with torch.no_grad():
                for name, model in models.items():
                    prediction = (
                        model(val_x)
                        if name == "mlp"
                        else model.sample(val_x, val_noise)
                    )
                    if args.validation_score == "energy":
                        samples = (
                            prediction[:, None]
                            if name == "mlp"
                            else model.sample(
                                val_x.repeat_interleave(16, 0),
                                score_noise.reshape(-1, 4),
                            ).reshape(len(val_x), 16, 4)
                        )
                        error = float(energy_score(samples, val_y))
                    else:
                        error = float((prediction - val_y).square().mean())
                    report["validation"][name] = error
                    if error < best[name]:
                        best[name] = error
                        checkpoint = dict(
                            schema=1,
                            method=name,
                            width=args.width,
                            state_dict={
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            },
                            normalization=normalization,
                            signature=metadata["signature"],
                            config=metadata["config"],
                            step=step,
                            validation_score=error,
                            validation_metric=args.validation_score,
                        )
                        check_budget(folder, 5_000_000)
                        torch.save(checkpoint, folder / (name + ".pt"))
            report["peak_gpu_bytes"] = (
                torch.cuda.max_memory_allocated() if device == "cuda" else 0
            )
            if report["peak_gpu_bytes"] > 10_000_000_000:
                raise RuntimeError("GPU allocation exceeded operator limit")
            print(json.dumps(report), flush=True)
            history.write(json.dumps(report) + "\n")
            history.flush()
    history.close()
    summary = dict(
        device=device,
        steps=args.steps,
        batch=args.batch,
        width=args.width,
        best_validation_score=best,
        validation_metric=args.validation_score,
        elapsed_s=time.perf_counter() - begun,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        artifact_bytes=check_budget(folder),
        motor_io=False,
    )
    (folder / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
