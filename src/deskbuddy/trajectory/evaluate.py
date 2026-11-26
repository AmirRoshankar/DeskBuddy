"""Held-out Cartesian planning benchmark with explicit validation/RRT fallback."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..learned.data import check_budget
from .collision import Scene
from .learning import Dense, Field, condition, decode_path, sample
from .rrt import resample, rrt_connect, shortcut


def classical(scene, case):
    rng = np.random.default_rng(100000 + case["index"])
    path = rrt_connect(scene, case["start"], case["goal"], rng)
    if path is None:
        return None
    try:
        path = resample(shortcut(scene, path, rng), 32)
    except ValueError:
        return None
    return path if scene.path_free(path) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/flowreach/trajectories-v1")
    parser.add_argument("--models", default="runs/flowreach/trajectories-v1/models-v3")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Need at least one held-out scenario")
    torch.set_num_threads(1)
    dataset, folder = Path(args.dataset), Path(args.models)
    check_budget(folder, 20_000_000)
    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    cases = [c for c in metadata["scenarios"] if c["split"] == "test"][: args.samples]
    models = {}
    checkpoints = {}
    for name in ("mlp", "fm", "mf", "imf"):
        cp = torch.load(folder / (name + ".pt"), map_location="cpu", weights_only=True)
        if cp.get("task") != "cartesian_trajectory_2d":
            raise ValueError("Wrong checkpoint task")
        model = (
            Dense(18, width=cp["width"])
            if name == "mlp"
            else Field(cp["width"], auxiliary=cp.get("auxiliary_velocity_head", False))
        )
        model.load_state_dict(cp["state_dict"])
        model.eval()
        models[name], checkpoints[name] = model, cp
    variants = [
        ("rrt", 0, 1),
        ("mlp", 1, 1),
        ("fm", 1, 1),
        ("fm", 4, 1),
        ("fm", 8, 1),
        ("fm", 16, 1),
        ("mf", 1, 1),
        ("mf", 4, 1),
        ("imf", 1, 1),
        ("imf", 4, 1),
        ("fm", 8, 4),
        ("mf", 1, 4),
        ("imf", 1, 4),
    ]
    summaries = []
    all_raw = []
    all_selected = []
    all_valid = []
    torch.manual_seed(1729)
    noise = torch.randn((len(cases), 4, 60))
    with torch.inference_mode():
        for model in models.values():
            for _ in range(10):
                sample(
                    model,
                    torch.zeros(1, 18),
                    torch.zeros(1, 60),
                    "imf" if isinstance(model, Field) else "mlp",
                )
        for name, steps, count in variants:
            label = f"{name}-{steps}nfe-{count}c"
            rows = []
            raw_paths = []
            selected_paths = []
            raw_valid = []
            for index, case in enumerate(cases):
                scene = Scene(**case["scene"])
                began = time.perf_counter()
                if name == "rrt":
                    output = classical(scene, case)
                    neural_ms = 0.0
                    validation_ms = 0.0
                    base_ms = (time.perf_counter() - began) * 1000
                    paths = [] if output is None else [output]
                    valid = [output] if output is not None else []
                else:
                    cp = checkpoints[name]
                    n = cp["normalization"]
                    cx = (
                        torch.tensor(
                            (condition(case) - n["c_mean"]) / n["c_std"],
                            dtype=torch.float32,
                        )
                        .reshape(1, -1)
                        .repeat(count, 1)
                    )
                    prediction = sample(
                        models[name], cx, noise[index, :count], name, steps
                    ).numpy()
                    residual = prediction * np.asarray(n["y_std"]) + n["y_mean"]
                    paths = [decode_path(p, case) for p in residual]
                    neural_ms = (time.perf_counter() - began) * 1000
                    validating = time.perf_counter()
                    valid = [p for p in paths if scene.path_free(p)]
                    validation_ms = (time.perf_counter() - validating) * 1000
                    base_ms = neural_ms + validation_ms
                    output = (
                        min(
                            valid,
                            key=lambda p: np.linalg.norm(
                                np.diff(p, axis=0), axis=1
                            ).sum(),
                        )
                        if valid
                        else None
                    )
                accepted = output is not None
                fallback_ms = 0.0
                if output is None and name != "rrt":
                    fallback_start = time.perf_counter()
                    output = classical(scene, case)
                    fallback_ms = (time.perf_counter() - fallback_start) * 1000
                row = dict(
                    scenario=case["index"],
                    blocked=case["direct_blocked"],
                    raw_valid=accepted,
                    final_valid=output is not None,
                    neural_ms=neural_ms,
                    validation_ms=validation_ms,
                    generation_validation_ms=base_ms,
                    fallback_ms=fallback_ms,
                    total_ms=base_ms + fallback_ms,
                )
                if accepted:
                    shortest = min(
                        valid,
                        key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum(),
                    )
                    row.update(
                        length_m=float(
                            np.linalg.norm(np.diff(shortest, axis=0), axis=1).sum()
                        ),
                        inflated_clearance_m=scene.clearance(shortest),
                        second_difference_m=float(
                            np.linalg.norm(
                                np.diff(shortest, n=2, axis=0), axis=1
                            ).mean()
                        ),
                    )
                rows.append(row)
                raw_paths.append(paths[0] if paths else np.full((32, 2), np.nan))
                selected_paths.append(
                    output if output is not None else np.full((32, 2), np.nan)
                )
                raw_valid.append(accepted)
            blocked = [r for r in rows if r["blocked"]]
            accepted_rows = [r for r in rows if r["raw_valid"]]
            result = dict(
                label=label,
                method=name,
                nfe_per_candidate=steps,
                candidates=count,
                sequential_field_calls=steps,
                sample_evaluations=steps * count,
                cases=len(rows),
                blocked_cases=len(blocked),
                raw_valid_rate=float(np.mean([r["raw_valid"] for r in rows])),
                blocked_raw_valid_rate=float(
                    np.mean([r["raw_valid"] for r in blocked])
                ),
                final_valid_rate=float(np.mean([r["final_valid"] for r in rows])),
                fallback_rate=float(np.mean([not r["raw_valid"] for r in rows]))
                if name != "rrt"
                else 0.0,
                median_length_m=float(np.median([r["length_m"] for r in accepted_rows]))
                if accepted_rows
                else None,
                median_inflated_clearance_m=float(
                    np.median([r["inflated_clearance_m"] for r in accepted_rows])
                )
                if accepted_rows
                else None,
            )
            for key in (
                "neural_ms",
                "validation_ms",
                "generation_validation_ms",
                "fallback_ms",
                "total_ms",
            ):
                result["median_" + key] = float(np.median([r[key] for r in rows]))
                result["p95_" + key] = float(np.percentile([r[key] for r in rows], 95))
            summaries.append(result)
            all_raw.append(raw_paths)
            all_selected.append(selected_paths)
            all_valid.append(raw_valid)
            print(json.dumps(result), flush=True)
    report = dict(
        schema=1,
        task="cartesian_trajectory_2d",
        synthetic_only=True,
        hardware_executable=False,
        models_directory=str(folder),
        test_scenario_ids=[c["index"] for c in cases],
        metrics=summaries,
        inference_device="CPU, one torch thread, warmed models, single pass per condition, preprocessing/decoding included",
        checkpoint_steps={k: v["step"] for k, v in checkpoints.items()},
        path_archive_layout={
            "raw": "first unfiltered candidate",
            "selected": "shortest valid candidate or validated RRT fallback",
            "valid": "at least one raw candidate valid",
        },
        limitations=[
            "No robot kinematics, full-arm geometry, perception, time parameterization or motor-energy estimates.",
            "Endpoints are fixed by construction, not learned.",
            "Raw validity is before fallback; final validity includes independently generated and checked RRT fallback.",
            "K-candidate policies increase sampling and validation cost; they do not make every generated path safe.",
            "Path length/clearance statistics only include accepted neural paths, so compare success rates alongside them.",
            "Collision checks use the same tested geometric proxy as teacher generation, not an independent physical measurement.",
        ],
    )
    (folder / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        folder / "evaluation-paths.npz",
        raw=np.array(all_raw, np.float32),
        selected=np.array(all_selected, np.float32),
        valid=np.array(all_valid),
        scenario_index=np.array(report["test_scenario_ids"]),
        labels=np.array([r["label"] for r in summaries]),
    )


if __name__ == "__main__":
    main()
