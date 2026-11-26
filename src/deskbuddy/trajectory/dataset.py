"""Generate a grouped 2D RRT teacher dataset and baseline timing report, offline."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..learned.data import check_budget
from .collision import Scene
from .rrt import resample, rrt_connect, shortcut


def layout(family, seed):
    rng = np.random.default_rng(seed + 1009 * family)
    center = rng.uniform([0.075, 0.09], [0.135, 0.205])
    radius = float(rng.uniform(0.012, 0.027))
    if family % 3 == 0:
        return Scene(circles=(tuple(center) + (radius,),))
    half = rng.uniform(0.012, 0.025, 2)
    box = tuple(center - half) + tuple(center + half)
    if family % 3 == 1:
        return Scene(boxes=(box,))
    second = [float(center[0]), float(np.clip(center[1] + 0.055, 0.04, 0.25)), 0.015]
    return Scene(circles=(tuple(second),), boxes=(box,))


def generate(folder, scenarios=5000, routes=2, seed=20261125):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "dataset.json").exists() or (folder / "dataset.npz").exists():
        raise ValueError("Choose a new output folder; existing datasets are preserved")
    if not 1 <= scenarios <= 50000 or not 1 <= routes <= 16:
        raise ValueError("Use 1–50000 scenarios and 1–16 route seeds")
    check_budget(folder, scenarios * routes * 4000 + 2_000_000)
    rng = np.random.default_rng(seed)
    scene_rows, attempts, paths, indices, splits = [], [], [], [], []
    began = time.perf_counter()
    for index in range(scenarios):
        family = index // 25
        split = (
            "test"
            if family % 10 == 0
            else "validation"
            if family % 10 == 1
            else "train"
        )
        scene = layout(family, seed)
        for _ in range(200):
            start = rng.uniform([0.012, 0.015], [0.07, 0.282])
            goal = rng.uniform([0.14, 0.015], [0.198, 0.282])
            if scene.segment_free(start, start) and scene.segment_free(goal, goal):
                break
        scene_rows.append(
            dict(
                index=index,
                family=family,
                split=split,
                scene=asdict(scene),
                start=start.tolist(),
                goal=goal.tolist(),
                direct_blocked=not scene.segment_free(start, goal),
            )
        )
        for route in range(routes):
            route_seed = seed + index * 101 + route
            planner_rng = np.random.default_rng(route_seed)
            started = time.perf_counter()
            path = rrt_connect(scene, start, goal, planner_rng)
            row = dict(
                scenario=index,
                family=family,
                split=split,
                route_seed=route_seed,
                status="rrt_failed",
                dataset_index=None,
            )
            if path is not None:
                path = shortcut(scene, path, planner_rng)
                try:
                    path = resample(path, 32).astype(np.float32)
                except ValueError:
                    path = None
                    row["status"] = "resampling_failed"
            row["generation_ms"] = (time.perf_counter() - started) * 1000
            if path is not None:
                validating = time.perf_counter()
                valid = (
                    scene.path_free(path)
                    and np.allclose(path[0], start, atol=1e-7, rtol=0)
                    and np.allclose(path[-1], goal, atol=1e-7, rtol=0)
                )
                row["validation_ms"] = (time.perf_counter() - validating) * 1000
                if valid:
                    row.update(
                        status="success",
                        dataset_index=len(paths),
                        length_m=float(
                            np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                        ),
                        inflated_clearance_m=scene.clearance(path),
                        second_difference_m=float(
                            np.linalg.norm(np.diff(path, n=2, axis=0), axis=1).mean()
                        ),
                    )
                    paths.append(path)
                    indices.append(index)
                    splits.append(split)
                else:
                    row["status"] = "validation_failed"
            row["total_ms"] = row["generation_ms"] + row.get("validation_ms", 0)
            attempts.append(row)
        if (index + 1) % 250 == 0:
            print(
                f"Generated {index + 1}/{scenarios} scenarios; {len(paths)}/{len(attempts)} valid trajectories",
                flush=True,
            )
    valid = [row for row in attempts if row["status"] == "success"]
    report = dict(
        schema=1,
        synthetic_only=True,
        hardware_executable=False,
        seed=seed,
        scenarios=scene_rows,
        attempts=attempts,
        representation="32 x 2 metres, approximately arc-uniform with all corners retained",
        split_method="25 start/goal cases per identical obstacle layout family; entire families and all route seeds stay in one split",
        metrics=dict(
            scenarios=scenarios,
            route_attempts=len(attempts),
            valid_trajectories=len(paths),
            success_rate=len(valid) / len(attempts),
            blocked_direct_fraction=sum(s["direct_blocked"] for s in scene_rows)
            / scenarios,
            median_generation_ms=float(
                np.median([r["generation_ms"] for r in attempts])
            ),
            median_validation_ms=float(
                np.median([r.get("validation_ms", 0) for r in attempts])
            ),
            median_total_ms=float(np.median([r["total_ms"] for r in attempts])),
            median_length_m=float(np.median([r["length_m"] for r in valid]))
            if valid
            else None,
            median_inflated_clearance_m=float(
                np.median([r["inflated_clearance_m"] for r in valid])
            )
            if valid
            else None,
            elapsed_s=time.perf_counter() - began,
        ),
        limitations=[
            "2D circular tool proxy; not an SO-101 joint or link collision test.",
            "No perception, IK, motor timing or learned trajectory model is included.",
            "Timing is CPU Python and includes shortcutting/resampling/validation, not scene creation, metric computation or file writes.",
            "Second difference is geometric, not physical acceleration or jerk; no time parameterization.",
            "Repeated seeds permit but do not guarantee different route modes. Failed attempts are retained.",
        ],
    )
    payload = json.dumps(report, indent=2)
    check_budget(folder, len(payload.encode()) + len(paths) * 32 * 2 * 4 + 1_000_000)
    np.savez_compressed(
        folder / "dataset.npz",
        paths=np.array(paths, np.float32).reshape(-1, 32, 2),
        scenario_index=np.array(indices, np.int32),
        split=np.array(splits),
    )
    (folder / "dataset.json").write_text(payload + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/flowreach/trajectories-v1")
    parser.add_argument("--scenarios", type=int, default=5000)
    parser.add_argument("--routes", type=int, default=2)
    args = parser.parse_args()
    generate(args.folder, args.scenarios, args.routes)


if __name__ == "__main__":
    main()
