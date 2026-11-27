"""AMF-inspired synthetic short-horizon pickup predictor. Offline; no motor APIs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..trajectory.learning import Dense
from .data import check_budget
from .grasp_lab import box_data

HORIZON = 16
CHANNELS = 6  # local XYZ / 50 mm, cos(yaw), sin(yaw), opening / 50 mm


def window_data(objects=1000, seed=20261126):
    contexts, poses, groups = box_data(objects, seed)
    rng = np.random.default_rng(seed + 1)
    observations, actions, ids = [], [], []
    for context, pose, identifier in zip(contexts, poses, groups):
        yaw = 2 * np.arctan2(pose[5], pose[4])
        start = np.r_[
            rng.uniform(-1, 1, 2),
            rng.uniform(1.5, 3),
            np.cos(yaw),
            np.sin(yaw),
            pose[7],
        ]
        above = np.r_[0.0, 0.0, 1.2, np.cos(yaw), np.sin(yaw), pose[7]]
        down = above.copy()
        down[2] = pose[2]
        closed = down.copy()
        closed[5] = max(0.0, pose[7] - 0.006 / 0.05)
        lifted = closed.copy()
        lifted[2] = 2.0
        episode = np.vstack(
            [
                np.linspace(start, above, 8),
                np.linspace(above, down, 14),
                np.linspace(down, closed, 7),
                np.linspace(closed, lifted, 19),
            ]
        )
        for phase in rng.integers(0, 44, 3):
            observations.append(
                np.r_[context, episode[phase], episode[max(0, phase - 1)]]
            )
            indices = np.minimum(
                np.arange(phase + 1, phase + 1 + HORIZON), len(episode) - 1
            )
            actions.append(episode[indices])
            ids.append(identifier)
    return (
        np.array(observations, np.float32),
        np.array(actions, np.float32),
        np.array(ids),
    )


class ActionField(torch.nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.net = Dense(
            17 + HORIZON * CHANNELS + HORIZON * 2, HORIZON * CHANNELS, width
        )

    def forward(self, z, r, t, context):
        value = torch.cat((context, z.flatten(1), r, t), -1)
        return z + self.net(value).reshape_as(z)


def intervals(batch, device):
    t, r = torch.rand(2, batch, HORIZON, device=device).sort(dim=0).values.flip(0)
    monotone = torch.rand(batch, 1, device=device) < 0.5
    near = torch.rand(batch, 1, device=device) * 0.2
    far = 0.6 + torch.rand(batch, 1, device=device) * 0.4
    ramp = torch.linspace(0, 1, HORIZON, device=device)[None]
    t = torch.where(monotone, near + (far - near) * ramp, t)
    r = torch.where(monotone, torch.zeros_like(r), r)
    r = torch.where(torch.rand(batch, 1, device=device) < 0.5, t, r)
    return r, t


def objective(model, data, context, method):
    r, t = intervals(len(data), data.device)
    noise = torch.randn_like(data)
    z = (1 - t[..., None]) * data + t[..., None] * noise
    velocity = noise - data

    def fn(z, r, t):
        return model(z, r, t, context)

    if method == "adaptive_imf":
        with torch.no_grad():
            tangent = model(z, t, t, context)
    else:
        tangent = velocity
    average, derivative = torch.func.jvp(
        fn, (z, r, t), (tangent, torch.zeros_like(r), torch.ones_like(t))
    )
    error = (
        average + (t - r)[..., None] * derivative.detach() - velocity
        if method == "adaptive_imf"
        else average - (velocity - (t - r)[..., None] * derivative).detach()
    )
    squared = error.square().mean((1, 2))
    return (squared / (squared.detach() + 0.001)).mean()


def replan(model, context, prior=None, near=0.08, far=0.85, noise=None):
    batch = len(context)
    device = context.device
    if prior is None:
        levels = torch.ones(batch, HORIZON, device=device)
        previous = torch.zeros(batch, HORIZON, CHANNELS, device=device)
    else:
        previous = torch.cat((prior[:, 2:], prior[:, -1:].expand(-1, 2, -1)), 1)
        levels = torch.linspace(near, far, HORIZON, device=device)[None].expand(
            batch, -1
        )
    if noise is None:
        noise = torch.randn_like(previous)
    z = (1 - levels[..., None]) * previous + levels[..., None] * noise
    return z - levels[..., None] * model(z, torch.zeros_like(levels), levels, context)


def train(folder, objects=1000, steps=5000, batch=256):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.pt")):
        raise ValueError("Use a fresh adaptive experiment folder")
    check_budget(folder, 50_000_000)
    torch.manual_seed(20261126)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 10_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
        )
    observations, actions, groups = window_data(objects)
    np.savez_compressed(
        folder / "dataset.npz",
        observations=observations,
        actions=actions,
        object_id=groups,
    )
    train_mask, test_mask = groups % 10 >= 2, groups % 10 == 0
    c_mean, c_std = (
        observations[train_mask].mean(0),
        np.maximum(observations[train_mask].std(0), 0.05),
    )
    y_mean, y_std = (
        actions[train_mask].mean((0, 1)),
        np.maximum(actions[train_mask].std((0, 1)), 0.05),
    )
    c = torch.tensor((observations - c_mean) / c_std, device=device)
    y = torch.tensor((actions - y_mean) / y_std, device=device)
    tx, ty, vx, vy = c[train_mask], y[train_mask], c[test_mask], y[test_mask]
    models = {
        name: ActionField().to(device) for name in ["adaptive_mf", "adaptive_imf"]
    }
    models["adaptive_imf"].load_state_dict(models["adaptive_mf"].state_dict())
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=3e-4)
        for name, model in models.items()
    }
    began = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(tx), (batch,), device=device)
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            value = objective(model, ty[indices], tx[indices], name)
            if not torch.isfinite(value):
                raise RuntimeError(f"Nonfinite {name} objective")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizers[name].step()
        if (step + 1) % 1000 == 0:
            print(
                json.dumps(dict(step=step + 1, elapsed_s=time.perf_counter() - began)),
                flush=True,
            )
    probe = vx[:100]
    mean, std = torch.tensor(y_mean, device=device), torch.tensor(y_std, device=device)
    reports = {}
    for name, model in models.items():
        metrics = {}
        with torch.inference_mode():
            first = replan(model, probe)
            shifted = torch.cat((first[:, 2:], first[:, -1:].expand(-1, 2, -1)), 1)
            for mode in ["fresh_noise", "reuse_previous"]:
                predicted = replan(
                    model, probe, first if mode == "reuse_previous" else None
                )
                physical = predicted * std + mean
                metrics[mode] = dict(
                    next_action_change_mm=float(
                        ((predicted[:, :4, :3] - shifted[:, :4, :3]) * std[:3] * 50)
                        .norm(dim=-1)
                        .mean()
                    ),
                    heldout_window_rmse=float(
                        (predicted - vy[: len(probe)]).square().mean().sqrt()
                    ),
                    below_plane_fraction=float((physical[:, :, 2] < 0).float().mean()),
                )
        reports[name] = metrics
        torch.save(
            dict(
                task="synthetic_adaptive_pickup",
                method=name,
                state_dict=model.cpu().state_dict(),
                normalization=dict(
                    c_mean=c_mean.tolist(),
                    c_std=c_std.tolist(),
                    y_mean=y_mean.tolist(),
                    y_std=y_std.tolist(),
                ),
                motor_io=False,
                physical_success_measured=False,
            ),
            folder / f"{name}.pt",
        )
    report = dict(
        objects=objects,
        windows=len(y),
        steps=steps,
        device=device,
        elapsed_s=time.perf_counter() - began,
        metrics=reports,
        motor_io=False,
        horizon=HORIZON,
        adaptation="Per-action interval conditioning and progressively noised shifted previous prediction; dense model, no image encoder",
        limitations=[
            "Procedural box approach/close/lift windows, not physical successful grasps.",
            "Consistency probe uses fixed observations; not a physical closed-loop or perturbation success benchmark.",
            "MF and iMF objectives are compared on the same windows; this is an adaptation, not a paper reproduction.",
            "Lower change alone can reward a frozen bad plan; check error and invalid poses as well.",
        ],
    )
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/deskbuddy/adaptive-box-grasps-v1")
    parser.add_argument("--objects", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    train(args.folder, args.objects, args.steps, args.batch)


if __name__ == "__main__":
    main()
