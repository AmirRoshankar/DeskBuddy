"""Offline synthetic box grasps: Euclidean baselines and a group-constrained flow map.

Geometry labels are procedural, not simulated contact successes. No hardware API.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..trajectory.learning import Dense, Field, loss, sample
from .data import check_budget


def unit(q):
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 1e-7, q / norm.clamp_min(1e-7), identity)


def multiply(a, b):
    w1, v1, w2, v2 = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat(
        (
            w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
            w1 * v2 + w2 * v1 + torch.linalg.cross(v1, v2),
        ),
        -1,
    )


def inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def rotation_log(q):
    q = unit(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., :1])
    return q[..., 1:] * (angle / length.clamp_min(1e-7))


def rotation_exp(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return unit(
        torch.cat(
            (torch.cos(angle / 2), vector * (0.5 * torch.sinc(angle / (2 * torch.pi)))),
            -1,
        )
    )


def pose_noise(batch, device):
    result = torch.randn(batch, 8, device=device)
    result[:, 3:7] = unit(result[:, 3:7])
    return result


def delta(a, b):
    return torch.cat(
        (
            b[:, :3] - a[:, :3],
            rotation_log(multiply(inverse(a[:, 3:7]), b[:, 3:7])),
            b[:, 7:] - a[:, 7:],
        ),
        -1,
    )


def advance(a, velocity, fraction):
    return torch.cat(
        (
            a[:, :3] + fraction * velocity[:, :3],
            multiply(a[:, 3:7], rotation_exp(fraction * velocity[:, 3:6])),
            a[:, 7:] + fraction * velocity[:, 6:],
        ),
        -1,
    )


class GraspMF(nn.Module):
    """Endpoint predictor; SO(3) is represented by normalized unit quaternions."""

    def __init__(self, width=128):
        super().__init__()
        self.net = Dense(15, 8, width)

    def endpoint(self, pose, s, t, context):
        output = self.net(torch.cat((context, pose, s, t), -1))
        rotation = unit(output[:, 3:7] + output.new_tensor([1.0, 0, 0, 0]))
        return torch.cat((output[:, :3], rotation, output[:, 7:]), -1)

    def velocity(self, pose, s, t, context):
        return delta(pose, self.endpoint(pose, s, t, context)) / (1 - s).clamp_min(0.05)

    def transport(self, pose, s, t, context):
        return advance(pose, self.velocity(pose, s, t, context), t - s)

    def objective(self, data, context):
        # Noise at s=0, data at s=1. This convention differs from trajectory/iMF.
        noise = pose_noise(len(data), data.device)
        s = torch.rand(len(data), 1, device=data.device) * 0.9
        pose = advance(noise, delta(noise, data), s)
        conditional = delta(pose, data) / (1 - s)
        anchor = (self.velocity(pose, s, s, context) - conditional).square().mean()
        t = s + 0.05 + torch.rand_like(s) * (1 - s - 0.05)
        middle = (s + t) / 2
        with torch.no_grad():
            halfway = self.transport(pose, s, middle, context)
            composed = self.transport(halfway, middle, t, context)
            target = delta(pose, composed) / (t - s)
        consistency = (self.velocity(pose, s, t, context) - target).square().mean()
        return anchor + 0.2 * consistency

    def draw(self, context, noise, steps=1):
        pose = noise
        for i in range(steps):
            s = pose.new_full((len(pose), 1), i / steps)
            t = pose.new_full((len(pose), 1), (i + 1) / steps)
            pose = self.transport(pose, s, t, context)
        return pose


def box_data(objects=2000, seed=20261126):
    rng = np.random.default_rng(seed)
    contexts, targets, groups = [], [], []
    for index in range(objects):
        size = rng.uniform([0.012, 0.012, 0.008], [0.04, 0.04, 0.035])
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        context = [*(size / 0.05), np.cos(2 * yaw), np.sin(2 * yaw)]
        for mode in range(2):
            angle = yaw + mode * np.pi / 2
            # Parallel jaws are symmetric under a 180-degree yaw.
            angle = (angle + np.pi / 2) % np.pi - np.pi / 2
            quat = [0.0, np.cos(angle / 2), np.sin(angle / 2), 0.0]
            target = [
                *rng.uniform(-0.02, 0.02, 2),
                size[2] * 0.55 / 0.05,
                *quat,
                (size[mode] + 0.006) / 0.05,
            ]
            contexts.append(context)
            targets.append(target)
            groups.append(index)
    return (
        np.array(contexts, np.float32),
        np.array(targets, np.float32),
        np.array(groups),
    )


def can_data(objects=1500, seed=20261126):
    rng = np.random.default_rng(seed)
    contexts = []
    targets = []
    groups = []
    for index in range(objects):
        diameter = rng.uniform(0.060, 0.074)
        height = rng.uniform(0.105, 0.145)
        for _ in range(4):
            yaw = rng.uniform(-np.pi / 2, np.pi / 2)
            contexts.append([diameter / 0.05, diameter / 0.05, height / 0.05, 1.0, 0.0])
            targets.append(
                [
                    *rng.uniform(-0.02, 0.02, 2),
                    height * 0.55 / 0.05,
                    0.0,
                    np.cos(yaw / 2),
                    np.sin(yaw / 2),
                    0.0,
                    (diameter + 0.006) / 0.05,
                ]
            )
            groups.append(index)
    return (
        np.array(contexts, np.float32),
        np.array(targets, np.float32),
        np.array(groups),
    )


def can_rim_data(objects=2000, seed=20261126):
    """Toy upright-can lip grasps; no contact simulation or hardware labels."""
    rng = np.random.default_rng(seed)
    contexts, targets, groups = [], [], []
    for index in range(objects):
        diameter = rng.uniform(0.060, 0.074)
        height = rng.uniform(0.115, 0.130)
        for _ in range(4):
            angle = rng.uniform(-np.pi, np.pi)
            yaw = (angle + np.pi / 2) % np.pi - np.pi / 2
            radial = diameter / 2 + rng.uniform(-0.0005, 0.0005)
            contexts.append(
                [
                    diameter / 0.05,
                    diameter / 0.05,
                    height / 0.05,
                    np.cos(angle),
                    np.sin(angle),
                ]
            )
            targets.append(
                [
                    radial * np.cos(angle) / 0.05,
                    radial * np.sin(angle) / 0.05,
                    (height - 0.002) / 0.05,
                    0.0,
                    np.cos(yaw / 2),
                    np.sin(yaw / 2),
                    0.0,
                    rng.uniform(0.007, 0.009) / 0.05,
                ]
            )
            groups.append(index)
    return (
        np.array(contexts, np.float32),
        np.array(targets, np.float32),
        np.array(groups),
    )


def geometric_metrics(prediction, context, return_mask=False, profile="box"):
    """A geometric label check, explicitly not a physics-based success metric."""
    p = prediction.detach().cpu().numpy()
    c = context.detach().cpu().numpy()
    q = unit(torch.tensor(p[:, 3:7])).numpy()
    # Rotation matrix x and z axes from scalar-first quaternion.
    w, x, y, z = q.T
    closing_x = 1 - 2 * (y * y + z * z)
    closing_y = 2 * (x * y + w * z)
    down = 1 - 2 * (x * x + y * y)
    yaw = np.arctan2(closing_y, closing_x)
    box_yaw = 0.5 * np.arctan2(c[:, 4], c[:, 3])
    difference = (yaw - box_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    mode = np.rint((yaw - box_yaw) / (np.pi / 2)).astype(int) % 2
    required = c[np.arange(len(c)), mode]
    xy_mm = np.linalg.norm(p[:, :2], axis=1) * 50
    if profile == "can_rim":
        radius = c[:, 0] * 0.05 / 2
        radial_error = np.abs(np.linalg.norm(p[:, :2] * 0.05, axis=1) - radius)
        height_error = np.abs(p[:, 2] * 0.05 - (c[:, 2] * 0.05 - 0.002))
        radial_angle = np.arctan2(p[:, 1], p[:, 0])
        angle_error = (yaw - radial_angle + np.pi / 2) % np.pi - np.pi / 2
        desired = np.arctan2(c[:, 4], c[:, 3])
        approach_error = (radial_angle - desired + np.pi) % (2 * np.pi) - np.pi
        valid = (
            (radial_error < 0.002)
            & (height_error < 0.002)
            & (down < -np.cos(np.deg2rad(12)))
            & (np.abs(angle_error) < np.deg2rad(15))
            & (np.abs(approach_error) < np.deg2rad(15))
            & (p[:, 7] * 0.05 >= 0.005)
            & (p[:, 7] * 0.05 <= 0.012)
        )
        if return_mask:
            return valid
        return dict(
            geometric_valid_rate=float(valid.mean()),
            mean_rim_radial_error_mm=float(radial_error.mean() * 1000),
            mean_rim_height_error_mm=float(height_error.mean() * 1000),
            downward_rate=float((down < -np.cos(np.deg2rad(12))).mean()),
        )
    valid = (
        (xy_mm < 3)
        & (p[:, 2] > 0.25 * c[:, 2])
        & (p[:, 2] < 0.85 * c[:, 2])
        & (down < -np.cos(np.deg2rad(12)))
        & (
            (abs(difference) < np.deg2rad(12))
            if profile == "box"
            else np.ones(len(p), bool)
        )
        & (p[:, 7] >= required + 0.002 / 0.05)
        & (p[:, 7] <= required + 0.012 / 0.05)
    )
    if return_mask:
        return valid
    return dict(
        geometric_valid_rate=float(valid.mean()),
        mean_center_error_mm=float(xy_mm.mean()),
        downward_rate=float((down < -np.cos(np.deg2rad(12))).mean()),
        face_alignment_rate=float((abs(difference) < np.deg2rad(12)).mean())
        if profile == "box"
        else None,
    )


def train(
    folder,
    objects=2000,
    steps=5000,
    batch=256,
    width=128,
    profile="box",
    physics_dataset=None,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.pt")):
        raise ValueError("Choose a fresh grasp experiment folder")
    check_budget(folder, 50_000_000)
    torch.manual_seed(20261126)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 10_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
        )
    physics_report = None
    if physics_dataset is not None:
        source = Path(physics_dataset)
        physics_report = json.loads(
            (source.parent / "report.json").read_text(encoding="utf-8")
        )
        if (
            physics_report.get("gravity_m_s2") != 9.81
            or physics_report.get("motor_io") is not False
        ):
            raise ValueError("Expected an offline gravity/contact physics dataset")
        with np.load(source) as archive:
            contexts, targets, groups = (
                archive[name].copy() for name in ("contexts", "targets", "object_id")
            )
        if len(targets) < 50 or len(set(groups[groups % 10 == 0])) < 5:
            raise ValueError(
                "Not enough independent successful physics examples/test objects"
            )
    else:
        contexts, targets, groups = (
            {"can": can_data, "can_rim": can_rim_data}.get(profile, box_data)
        )(objects)
    np.savez_compressed(
        folder / "dataset.npz", contexts=contexts, targets=targets, object_id=groups
    )
    c, y = torch.tensor(contexts, device=device), torch.tensor(targets, device=device)
    train_mask, test_mask = groups % 10 >= 2, groups % 10 == 0
    tx, ty, vx = c[train_mask], y[train_mask], c[test_mask]
    models = dict(
        mlp=Dense(5, 8, width),
        fm=Field(width, condition_dim=5, state_dim=8),
        imf=Field(width, condition_dim=5, state_dim=8),
        grasp_mf=GraspMF(width),
    )
    if profile == "can_rim":
        models = {"imf": models["imf"]}
    models = {name: model.to(device) for name, model in models.items()}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=3e-4)
        for name, model in models.items()
    }
    began = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(len(tx), (batch,), device=device)
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            value = (
                model.objective(ty[indices], tx[indices])
                if name == "grasp_mf"
                else loss(
                    model,
                    tx[indices],
                    ty[indices],
                    name,
                    warmup=step < min(500, steps // 10),
                    diagonal_fraction=0.5,
                )
            )
            if not torch.isfinite(value):
                raise RuntimeError(f"Nonfinite {name} loss")
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizers[name].step()
        if (step + 1) % 1000 == 0:
            print(
                json.dumps(dict(step=step + 1, elapsed_s=time.perf_counter() - began)),
                flush=True,
            )
    reports = {}
    generator = torch.Generator(device=device).manual_seed(1739)
    euclidean = torch.randn((4, len(vx), 8), device=device, generator=generator)
    group_noise = torch.stack([pose_noise(len(vx), device) for _ in range(4)])
    for name, model in models.items():
        with torch.inference_mode():
            guesses = [
                model.draw(vx, z)
                if name == "grasp_mf"
                else sample(model, vx, z, name, 8 if name == "fm" else 1)
                for z in (
                    group_noise
                    if name == "grasp_mf"
                    else euclidean[:1]
                    if name == "mlp"
                    else euclidean
                )
            ]
        reports[name] = [geometric_metrics(p, vx, profile=profile) for p in guesses]
        torch.save(
            dict(
                task="synthetic_can_grasp"
                if profile in ("can", "can_rim")
                else "synthetic_box_grasp",
                profile=profile,
                method=name,
                width=width,
                state_dict=model.cpu().state_dict(),
                input_dim=5,
                output_dim=8,
                motor_io=False,
                physical_grasp_success_measured=False,
                physics_training=physics_report is not None,
                physics_engine=physics_report["engine"]
                if physics_report
                else "none; geometric labels only",
            ),
            folder / f"{name}.pt",
        )
    report = dict(
        profile=profile,
        objects=objects,
        examples=len(y),
        steps=steps,
        batch=batch,
        width=width,
        device=device,
        elapsed_s=time.perf_counter() - began,
        metrics=reports,
        motor_io=False,
        physics_training=physics_report,
        representation="object-relative XYZ / 50 mm, scalar-first quaternion, opening / 50 mm",
        context="can XYZ size / 50 mm; cos(radial approach), sin(radial approach)"
        if profile == "can_rim"
        else "box XYZ size / 50 mm; cos(2*yaw), sin(2*yaw)",
        split="object id, all pose modes stay in one split",
        limitations=[
            "Toy MuJoCo contact/gravity-filtered labels; no measured real-robot success."
            if physics_report
            else "Procedural geometry labels, no contact dynamics or physical success.",
            "Fixed downward tabletop grasps, not an ACRONYM reproduction.",
            "No evidence of transfer to keys, chains or live robot control.",
        ],
    )
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/deskbuddy/box-grasps-v1")
    parser.add_argument("--objects", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--profile", choices=["box", "can", "can_rim"], default="box")
    parser.add_argument(
        "--physics-dataset",
        help="Successful MuJoCo dataset.npz, with adjacent report.json",
    )
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.folder)
    else:
        train(
            args.folder,
            args.objects,
            args.steps,
            args.batch,
            profile=args.profile,
            physics_dataset=args.physics_dataset,
        )


def evaluate(folder):
    folder = Path(folder)
    data = np.load(folder / "dataset.npz")
    test = data["object_id"] % 10 == 0
    c = torch.tensor(data["contexts"][test])
    torch.manual_seed(1739)
    torch.set_num_threads(1)
    summary, predictions = {}, {}
    for name, method, steps in [
        ("mlp", "mlp", 1),
        ("fm", "fm", 8),
        ("imf", "imf", 1),
        ("grasp_mf", "grasp_mf", 1),
        ("grasp_mf_5", "grasp_mf", 5),
    ]:
        if not (folder / f"{method}.pt").exists():
            continue
        cp = torch.load(folder / f"{method}.pt", map_location="cpu", weights_only=True)
        model = (
            GraspMF(cp["width"])
            if method == "grasp_mf"
            else Dense(5, 8, cp["width"])
            if method == "mlp"
            else Field(cp["width"], condition_dim=5, state_dim=8)
        )
        model.load_state_dict(cp["state_dict"])
        model.eval()
        draws = []
        began = time.perf_counter()
        with torch.inference_mode():
            for _ in range(1 if method == "mlp" else 4):
                noise = (
                    pose_noise(len(c), "cpu")
                    if method == "grasp_mf"
                    else torch.randn(len(c), 8)
                )
                draws.append(
                    model.draw(c, noise, steps)
                    if method == "grasp_mf"
                    else sample(model, c, noise, method, steps)
                )
        generation_ms = (time.perf_counter() - began) * 1000 / len(c)
        masks = np.array(
            [
                geometric_metrics(p, c, True, profile=cp.get("profile", "box"))
                for p in draws
            ]
        )
        summary[name] = dict(
            nfe=steps,
            candidates=len(draws),
            geometric_valid_one=float(masks[0].mean()),
            geometric_valid_any=float(masks.any(0).mean()),
            batched_generation_ms_per_condition=generation_ms,
            physical_success_measured=False,
        )
        predictions[name] = torch.stack(draws).numpy()
    np.savez_compressed(
        folder / "heldout_predictions.npz", contexts=c.numpy(), **predictions
    )
    (folder / "evaluation.json").write_text(
        json.dumps(
            dict(
                summary=summary,
                motor_io=False,
                metric="procedural grasp geometry criteria, not physics or physical pickup success",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
