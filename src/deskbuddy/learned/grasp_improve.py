"""Offline can-rim grasp improvement: success scorer, reward fine-tuning, body baseline.

Everything here runs in the headless MuJoCo toy simulator (can_physics); it
never imports hardware code or commands motors. Three ideas from recent work:

* Success scorer (GraspGen-style generator + discriminator). An MLP predicts
  P(pick, transfer, drop succeeds | can context, grasp) from *all* physics
  rollouts, failures included, plus rollouts of the generator's own samples
  ("on-generator" data, so the scorer learns this generator's mistakes).
  At inference iMF draws K grasps and the scorer picks one: still one NFE per
  draw, K draws in one batch.
* Reward fine-tuning (filtered behaviour cloning, the simplest policy
  improvement for a binary reward). Each round samples grasps from the
  current one-step iMF on training cans, simulates them, and fine-tunes on
  every success found so far. It is the bandit special case of the RL
  fine-tuning used by DMPO/ReinFlow, without a value function.
* Numerical body-grasp baseline in the same simulator, so the rim strategy
  can be compared with the simplest alternative under identical physics.

All simulation uses the *necked* toy can (66 mm body, 54.1 mm 202 lid); the
earlier straight-walled datasets put the lip ~6 mm too far out, so they are
not reused. A fresh dataset is generated, then split by can identity:
object_id % 10 == 0 is held out, % 10 == 1 validates the scorer, the rest trains.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Earlier checkpoints, evaluated as deployed: body-radius output moved onto the
# lid seam exactly like can_discard.to_lid_radius does live.
LEGACY = dict(
    live_geometric="runs/deskbuddy/can-rim-grasps-v1/imf.pt",
    physics_v1="runs/deskbuddy/can-rim-grasps-physics-v1/imf.pt",
)
SCALE = 0.05
LID = 0.0541


# --------------------------------------------------------------------------
# Data


def generate_rows(objects, attempts, seed):
    """Broad rim proposals around the lid seam for fresh necked-can rollouts."""
    rng = np.random.default_rng(seed)
    contexts, targets, ids, params = [], [], [], []
    for index in range(objects):
        diameter = rng.uniform(0.060, 0.074)
        height = rng.uniform(0.115, 0.130)
        mass = rng.uniform(0.015, 0.4)
        friction = rng.uniform(0.35, 1.1)
        for _ in range(attempts):
            angle = rng.uniform(-np.pi, np.pi)
            yaw = (angle + rng.normal(0, np.deg2rad(6)) + np.pi / 2) % np.pi - np.pi / 2
            radius = LID / 2 + rng.uniform(-0.003, 0.003)
            depth = rng.uniform(0.0005, 0.006)
            offset = [radius * np.cos(angle), radius * np.sin(angle), height - depth]
            contexts.append(
                [
                    diameter / SCALE,
                    diameter / SCALE,
                    height / SCALE,
                    np.cos(angle),
                    np.sin(angle),
                ]
            )
            targets.append(
                [
                    *(np.asarray(offset) / SCALE),
                    0.0,
                    np.cos(yaw / 2),
                    np.sin(yaw / 2),
                    0.0,
                    rng.uniform(0.006, 0.014) / SCALE,
                ]
            )
            ids.append(index)
            params.append([mass, friction, 4.0])
    return dict(
        contexts=np.float32(contexts),
        targets=np.float32(targets),
        object_id=np.asarray(ids),
        params=np.float32(params),
    )


def to_lid(grasps, contexts):
    """Deployed correction for body-radius models (see can_discard.to_lid_radius)."""
    out = np.array(grasps, copy=True)
    xy = out[..., :2] * SCALE
    radial = np.linalg.norm(xy, axis=-1, keepdims=True)
    shift = np.asarray(contexts)[..., :1] * SCALE / 2 - LID / 2
    out[..., :2] = (
        xy * np.maximum(radial - shift, 1e-6) / np.maximum(radial, 1e-9) / SCALE
    )
    return out


def split(object_id):
    fold = np.asarray(object_id) % 10
    return dict(test=fold == 0, val=fold == 1, train=fold >= 2)


def evaluation_rows(data):
    """Every v1-style held-out row: one context per (can, direction)."""
    test = split(data["object_id"])["test"]
    rows, seen = [], set()
    for index in np.flatnonzero(test):
        key = (int(data["object_id"][index]), *np.round(data["contexts"][index], 5))
        if key not in seen:
            seen.add(key)
            rows.append(index)
    return np.asarray(rows)


def wilson(successes, total, z=1.96):
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


# --------------------------------------------------------------------------
# Simulation (parallel worker processes; Windows-safe module-level functions)


def _simulate_chunk(rows):
    from .can_physics import body_rollout, rollout

    results = []
    for kind, context, grasp, params in rows:
        diameter, height = float(context[0] * SCALE), float(context[2] * SCALE)
        mass, friction, force = (float(v) for v in params)
        if kind == "body":
            yaw = float(np.arctan2(context[4], context[3]) + np.pi / 2)
            outcome = body_rollout(
                diameter, height, yaw, mass, friction, force, lid_diameter=LID
            )
        else:
            grasp = np.asarray(grasp, float)
            opening = float(grasp[7] * SCALE)
            if not np.isfinite(grasp).all() or not 0.002 < opening < 0.025:
                outcome = dict(success=False, pickup=False, invalid_output=True)
            else:
                outcome = rollout(
                    diameter,
                    height,
                    grasp[:3] * SCALE,
                    grasp[3:7],
                    opening,
                    mass=mass,
                    friction=friction,
                    force=force,
                    lid_diameter=LID,
                )
        results.append(
            dict(
                success=bool(outcome["success"]),
                pickup=bool(outcome["pickup"]),
                invalid=bool(outcome.get("invalid_output", False)),
            )
        )
    return results


def simulate(rows, workers):
    """rows: list of (kind, context, grasp, params); returns outcome dicts."""
    rows = [
        (kind, np.asarray(c, np.float64), None if g is None else np.asarray(g), p)
        for kind, c, g, p in rows
    ]
    if workers <= 1 or len(rows) < 64:
        return _simulate_chunk(rows)
    size = max(16, len(rows) // (workers * 4))
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [item for part in pool.map(_simulate_chunk, chunks) for item in part]


# --------------------------------------------------------------------------
# Models


def load_generator(path):
    import torch

    from ..trajectory.learning import Field

    cp = torch.load(path, map_location="cpu", weights_only=True)
    model = Field(cp["width"], condition_dim=5, state_dim=8)
    model.load_state_dict(cp["state_dict"])
    model.eval()
    return model, cp


def draw(model, contexts, count, seed, device="cpu"):
    """count one-step iMF grasps per context -> (count, N, 8) numpy."""
    import torch

    from ..trajectory.learning import sample
    from .grasp_lab import unit

    c = torch.as_tensor(np.asarray(contexts, np.float32), device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((count, len(c), 8), generator=generator).to(device)
    model = model.to(device)
    with torch.inference_mode():
        out = torch.stack([sample(model, c, z, "imf", 1) for z in noise])
        out[..., 3:7] = unit(out[..., 3:7])
    return out.cpu().numpy()


class Scorer:
    """MLP success classifier on (context, grasp), standardized inputs."""

    def __init__(self, width=128):
        from ..trajectory.learning import Dense

        self.net = Dense(13, 1, width)
        self.mean = np.zeros(13, np.float32)
        self.std = np.ones(13, np.float32)

    def features(self, contexts, grasps):
        x = np.concatenate([np.asarray(contexts), np.asarray(grasps)], -1)
        return ((x - self.mean) / self.std).astype(np.float32)

    def fit(self, contexts, grasps, labels, steps=6000, batch=1024, device="cpu"):
        import torch

        x = np.concatenate([contexts, grasps], -1).astype(np.float32)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        x = torch.as_tensor(self.features(contexts, grasps), device=device)
        y = torch.as_tensor(np.asarray(labels, np.float32), device=device)
        positives = float(y.sum())
        weight = torch.tensor(
            min(50.0, (len(y) - positives) / max(positives, 1.0)), device=device
        )
        net = self.net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(11)
        for _ in range(steps):
            index = torch.randint(len(x), (batch,), generator=generator).to(device)
            logits = net(x[index]).squeeze(-1)
            value = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[index], pos_weight=weight
            )
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        self.net = net.cpu().eval()
        return self

    def score(self, contexts, grasps):
        import torch

        shape = np.asarray(grasps).shape[:-1]
        c = np.broadcast_to(np.asarray(contexts), shape + (5,)).reshape(-1, 5)
        g = np.asarray(grasps).reshape(-1, 8)
        with torch.inference_mode():
            logits = self.net(torch.as_tensor(self.features(c, g))).squeeze(-1)
        return logits.numpy().reshape(shape)

    def state(self):
        return dict(
            state_dict=self.net.state_dict(),
            mean=self.mean.tolist(),
            std=self.std.tolist(),
            input="context(5) + grasp(8), standardized",
        )


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def fine_tune(
    model, contexts, grasps, steps, device="cpu", lr=1e-4, batch=256, warmup_steps=0
):
    import torch
    from torch import nn

    from ..trajectory.learning import loss

    model = model.to(device).train()
    c = torch.as_tensor(np.asarray(contexts, np.float32), device=device)
    y = torch.as_tensor(np.asarray(grasps, np.float32), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(29)
    for step in range(steps):
        index = torch.randint(len(c), (batch,), generator=generator).to(device)
        value = loss(
            model,
            c[index],
            y[index],
            "imf",
            warmup=step < warmup_steps,
            diagonal_fraction=0.5,
        )
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite fine-tuning loss")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return model.cpu().eval()


# --------------------------------------------------------------------------
# Experiment


def run(
    folder,
    objects=3000,
    attempts=8,
    rounds=4,
    train_objects=600,
    draws_per_context=4,
    best_of=16,
    base_steps=10000,
    steps_per_round=3000,
    workers=None,
    smoke=False,
):
    import torch

    from ..trajectory.learning import Field
    from .data import check_budget

    folder = Path(folder)
    if (folder / "report.json").exists():
        raise ValueError("Choose a fresh experiment folder")
    folder.mkdir(parents=True, exist_ok=True)
    check_budget(folder, 300_000_000)
    torch.manual_seed(20261127)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    if smoke:
        objects, attempts, rounds, train_objects = 150, 8, 1, 12
        best_of, base_steps, steps_per_round = 4, 60, 30
    began = time.perf_counter()
    log = []

    def note(**values):
        values["elapsed_s"] = round(time.perf_counter() - began, 1)
        log.append(values)
        print(json.dumps(values), flush=True)

    note(stage="start", device=device, workers=workers, objects=objects)

    # 1. Fresh necked-can dataset: every rollout kept, failures included.
    data = generate_rows(objects, attempts, seed=20261127)
    outcomes = simulate(
        [
            ("rim", c, g, p)
            for c, g, p in zip(data["contexts"], data["targets"], data["params"])
        ],
        workers,
    )
    data["success"] = np.array([o["success"] for o in outcomes])
    data["pickup"] = np.array([o["pickup"] for o in outcomes])
    np.savez_compressed(folder / "necked-rollouts.npz", **data)
    note(
        stage="dataset",
        rollouts=len(outcomes),
        success_rate=float(data["success"].mean()),
        pickup_rate=float(data["pickup"].mean()),
    )
    parts = split(data["object_id"])
    pos = parts["train"] & data["success"]
    if pos.sum() < (1 if smoke else 50):
        raise RuntimeError(
            f"Only {int(pos.sum())} successful training rollouts; widen sampling"
        )

    # 2. iMF from scratch on successful necked-can rollouts (same objective).
    base = Field(128, condition_dim=5, state_dim=8)
    base = fine_tune(
        base,
        data["contexts"][pos],
        data["targets"][pos],
        base_steps,
        device,
        lr=3e-4,
        warmup_steps=min(500, base_steps // 10),
    )
    note(stage="base_model", positives=int(pos.sum()))
    template = load_generator(LEGACY["live_geometric"])[1]

    rng = np.random.default_rng(5)
    train_ids = np.unique(data["object_id"][parts["train"]])
    chosen = rng.choice(train_ids, min(train_objects, len(train_ids)), replace=False)
    round_rows = np.concatenate(
        [
            rng.choice(np.flatnonzero(data["object_id"] == i), 2, replace=False)
            for i in chosen
        ]
    )

    def on_generator(model, seed):
        c = data["contexts"][round_rows]
        grasps = draw(model, c, draws_per_context, seed, device)
        rows = [
            ("rim", c[j], grasps[k, j], data["params"][round_rows[j]])
            for k in range(draws_per_context)
            for j in range(len(c))
        ]
        results = simulate(rows, workers)
        labels = np.array([o["success"] for o in results])
        pickups = float(np.mean([o["pickup"] for o in results]))
        return (
            np.concatenate([c] * draws_per_context),
            grasps.reshape(-1, 8),
            labels,
            pickups,
        )

    scorer_c = [data["contexts"][~parts["test"]]]
    scorer_g = [data["targets"][~parts["test"]]]
    scorer_y = [data["success"][~parts["test"]]]
    scorer_split = [np.where(parts["val"][~parts["test"]], "val", "train")]
    pos_c, pos_g = data["contexts"][pos], data["targets"][pos]

    # 3. Reward fine-tuning: sample, simulate, keep successes, fine-tune.
    improved = copy_model(base)
    history = []
    for index in range(rounds):
        c, g, y, pickups = on_generator(improved, 1000 + index)
        pos_c, pos_g = np.concatenate([pos_c, c[y]]), np.concatenate([pos_g, g[y]])
        for bucket, value in ((scorer_c, c), (scorer_g, g), (scorer_y, y)):
            bucket.append(value)
        scorer_split.append(np.full(len(y), "train"))
        history.append(
            dict(
                round=index + 1,
                sampled_success_rate=float(y.mean()),
                sampled_pickup_rate=pickups,
                positives=len(pos_c),
            )
        )
        note(stage="fine_tune_round", **history[-1])
        improved = fine_tune(improved, pos_c, pos_g, steps_per_round, device)

    # 4. Success scorer on everything except held-out cans.
    sc_c, sc_g = np.concatenate(scorer_c), np.concatenate(scorer_g)
    sc_y, sc_s = np.concatenate(scorer_y), np.concatenate(scorer_split)
    scorer = Scorer().fit(
        sc_c[sc_s == "train"],
        sc_g[sc_s == "train"],
        sc_y[sc_s == "train"],
        steps=200 if smoke else 6000,
        device=device,
    )
    val_auc = auc(
        scorer.score(sc_c[sc_s == "val"], sc_g[sc_s == "val"]), sc_y[sc_s == "val"]
    )
    note(stage="scorer", examples=int((sc_s == "train").sum()), val_auc=val_auc)

    # 5. Held-out cans: identical contexts, physics parameters and noise.
    rows = evaluation_rows(data)
    if smoke:
        rows = rows[:24]
    contexts, params = data["contexts"][rows], data["params"][rows]
    methods = {}

    def evaluate(name, grasps=None, kind="rim", extra=None):
        results = simulate(
            [
                (kind, contexts[j], None if grasps is None else grasps[j], params[j])
                for j in range(len(contexts))
            ],
            workers,
        )
        n = len(results)
        success = sum(o["success"] for o in results)
        pickup = sum(o["pickup"] for o in results)
        methods[name] = dict(
            contexts=n,
            full_cycle_successes=int(success),
            full_cycle_rate=success / n,
            full_cycle_95ci=wilson(success, n),
            pickups=int(pickup),
            pickup_rate=pickup / n,
            pickup_95ci=wilson(pickup, n),
            invalid_outputs=int(sum(o["invalid"] for o in results)),
            **(extra or {}),
        )
        note(stage="evaluate", method=name, **methods[name])

    candidates = [
        (name, load_generator(path)[0], True) for name, path in LEGACY.items()
    ] + [("necked_imf", base, False), ("necked_imf_reward_finetuned", improved, False)]
    for name, model, legacy in candidates:
        one = draw(model, contexts, 1, 4242, device)[0]
        many = draw(model, contexts, best_of, 4242, device)
        if legacy:
            one, many = to_lid(one, contexts), to_lid(many, contexts[None])
        evaluate(f"{name}_1draw", one, extra=dict(nfe=1, draws=1, lid_corrected=legacy))
        pick = many[
            scorer.score(contexts[None], many).argmax(0), np.arange(len(contexts))
        ]
        evaluate(
            f"{name}_best_of_{best_of}_scored",
            pick,
            extra=dict(
                nfe=1, draws=best_of, selection="success scorer", lid_corrected=legacy
            ),
        )
    evaluate("numerical_body_baseline", kind="body", extra=dict(nfe=0))

    common = dict(
        template,
        physics_training=True,
        physics_geometry="necked 211/202 toy can, lid 54.1 mm",
        physical_grasp_success_measured=False,
        motor_io=False,
    )
    torch.save(
        dict(common, state_dict=base.cpu().state_dict()), folder / "imf-necked.pt"
    )
    torch.save(
        dict(
            common,
            state_dict=improved.cpu().state_dict(),
            reward_finetuning=dict(
                method="filtered behaviour cloning on simulated successes",
                rounds=rounds,
                history=history,
            ),
        ),
        folder / "imf-necked-reward-finetuned.pt",
    )
    torch.save(dict(scorer.state(), motor_io=False), folder / "success-scorer.pt")
    report = dict(
        purpose="Offline toy-physics comparison; not physical robot evidence.",
        motor_io=False,
        full_robot_simulated=False,
        smoke=smoke,
        device=device,
        geometry=dict(
            body="60-74 mm", lid_mm=LID * 1000, neck_mm=19, seam="2.6 mm ring"
        ),
        split="object_id % 10: 0 held out, 1 scorer validation, else train",
        dataset=dict(
            objects=objects,
            attempts=attempts,
            rollouts=int(len(data["success"])),
            success_rate=float(data["success"].mean()),
            pickup_rate=float(data["pickup"].mean()),
        ),
        evaluation_contexts=int(len(rows)),
        methods=methods,
        fine_tune_history=history,
        scorer=dict(val_auc=val_auc, examples=int((sc_s == "train").sum())),
        limitations=[
            "Rigid toy can, mocap hand and box fingers; SO-101 arm, servos and real jaw not simulated.",
            "Pour-hole grasp is not simulated (no opened lid shell).",
            "Body baseline uses assumed 4x20x40 mm pads, not a measured SO-101 jaw.",
            "Nothing is promoted to the live controller automatically.",
        ],
        log=log,
        elapsed_s=time.perf_counter() - began,
    )
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def copy_model(model):
    import copy

    return copy.deepcopy(model).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--objects", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--best-of", type=int, default=16)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(
        args.folder,
        objects=args.objects,
        rounds=args.rounds,
        best_of=args.best_of,
        workers=args.workers,
        smoke=args.smoke,
    )
    summary = {
        name: dict(
            full_cycle=f"{m['full_cycle_successes']}/{m['contexts']}",
            pickup=f"{m['pickups']}/{m['contexts']}",
        )
        for name, m in report["methods"].items()
    }
    print(
        json.dumps(dict(methods=summary, scorer=report["scorer"]), indent=2), flush=True
    )


if __name__ == "__main__":
    main()
