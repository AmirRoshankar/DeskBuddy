"""Offline, shared full-arm RRT teachers. This module has no hardware entry point."""

import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import load_config
from ..kinematics import Arm
from ..learned.data import check_budget
from .execution import check_swept_obstacles
from .joint_planner import JointSpace
from .rrt import rrt_connect, shortcut

MAX_BOXES = 6
METHODS = ("mlp", "fm", "imf")


def condition(case, limits):
    """Robot dimensions stay fixed; all geometry is expressed in the base frame."""
    low, high = np.asarray(limits)
    cfg = case["config"]
    normal = np.asarray(case["table_normal"])
    rows = []
    for box in sorted(case["boxes"], key=lambda b: tuple(b["low_m"])):
        a, b = np.array(box["low_m"]), np.array(box["high_m"])
        rows.append([1.0, *((a + b) / 2 / 0.5), *((b - a) / 2 / 0.5)])
    if len(rows) > MAX_BOXES:
        raise ValueError("Full-arm experiment supports up to six boxes")
    return np.r_[
        2 * (np.asarray(case["start"]) - low) / (high - low) - 1,
        2 * (np.asarray(case["goal"]) - low) / (high - low) - 1,
        np.array(cfg["workspace_min_m"]) / 0.5,
        np.array(cfg["workspace_max_m"]) / 0.5,
        normal,
        case["plane_offset_m"] / 0.5,
        np.array(rows + [[0.0] * 7] * (MAX_BOXES - len(rows))).reshape(-1),
        [
            name in cfg["locked_joints"]
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            )
        ],
    ].astype(np.float32)


def encode_path(path, case, limits):
    line = np.linspace(case["start"], case["goal"], 32)
    return ((path[1:-1] - line[1:-1]) / (np.asarray(limits)[1] - limits[0])).reshape(-1)


def decode_path(residual, case, limits):
    path = np.linspace(case["start"], case["goal"], 32)
    path[1:-1] += np.asarray(residual).reshape(30, 5) * (
        np.asarray(limits)[1] - limits[0]
    )
    return path


def resample_joint(path, count=32):
    path = np.asarray(path)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12]]
    if not 2 <= len(path) <= count:
        raise ValueError("Cannot retain every teacher corner in 32 slots")
    length = np.linalg.norm(np.diff(path, axis=0), axis=1)
    intervals = np.ones(len(length), int)
    for _ in range(count - len(path)):
        intervals[np.argmax(length / intervals)] += 1
    points = [path[0]]
    for a, b, n in zip(path[:-1], path[1:], intervals):
        points.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(points)


def layout(base, family, seed):
    rng = np.random.default_rng(seed + 1009 * family)
    cfg = copy.deepcopy(base)
    cfg.update(allow_object_collisions=False, restrict_to_test_bed=False)
    cfg["locked_joints"] = ["wrist_roll"] if family % 2 else []
    cfg["workspace_min_m"] = rng.uniform(
        [0.04, -0.30, 0.015], [0.10, -0.20, 0.045]
    ).tolist()
    cfg["workspace_max_m"] = rng.uniform(
        [0.36, 0.20, 0.25], [0.45, 0.30, 0.37]
    ).tolist()
    rotation = Rotation.from_euler(
        "xy", rng.uniform(-5, 5, 2), degrees=True
    ).as_matrix()
    origin = np.array([0.12, -0.1485, rng.uniform(-0.008, 0.008)])
    bed = np.array([[0, 0, 0], [0.21, 0, 0], [0.21, 0.297, 0], [0, 0.297, 0]])
    cfg["test_bed"] = dict(
        corners_base_m=(bed @ rotation.T + origin).tolist(),
        inset_m=0.005,
        clearance_margin_m=0.0,
    )
    normal = rotation[:, 2]
    offset = -normal @ origin
    boxes = []
    # Seven density strata, with table objects, thin walls, tall columns and raised boxes.
    for index in range(family % 7):
        xy = rng.uniform([0.14, -0.23], [0.39, 0.23])
        half = rng.uniform(0.008, 0.045, 2)
        if family % 4 == 1:
            half[index % 2] = rng.uniform(0.004, 0.009)
            half[1 - index % 2] = rng.uniform(0.045, 0.09)
        z = float(-(normal[:2] @ xy + offset) / normal[2])
        bottom = z + (rng.uniform(0.04, 0.14) if family % 4 == 3 else 0.0)
        height = float(rng.uniform(0.015, 0.20))
        boxes.append(
            dict(
                low_m=[*(xy - half), bottom],
                high_m=[*(xy + half), bottom + height],
                pixels=1,
                height_above_table_m=bottom + height - z,
                pixel_roi=[0, 0, 1, 1],
            )
        )
    return cfg, boxes, normal, float(offset)


def generate_family(arguments):
    base, family, queries, routes, seed = arguments
    cfg, boxes, normal, offset = layout(base, family, seed)
    arm = Arm(cfg["model_joint_limits_deg"], cfg["tool_offset_m"])
    rng = np.random.default_rng(seed + family * 7919)
    split = (
        "test" if family % 10 == 0 else "validation" if family % 10 == 1 else "train"
    )
    cases, attempts, paths = [], [], []
    for query in range(queries):
        start = goal = None
        checker = JointSpace(arm, cfg, boxes, np.zeros(5), seconds=60)
        # Locks apply to a trajectory, not to sampling its initial pose.
        checker.low, checker.high = arm.lower.copy(), arm.upper.copy()
        for _ in range(1500):
            q = rng.uniform(arm.lower + 0.01, arm.upper - 0.01)
            if start is not None and cfg["locked_joints"]:
                q[4] = start[4]
            if checker.valid(q):
                if start is None:
                    start = q
                elif np.linalg.norm(q - start) > 0.4:
                    goal = q
                    break
        case = dict(
            family=family,
            query=query,
            split=split,
            config=cfg,
            boxes=boxes,
            table_normal=normal.tolist(),
            plane_offset_m=offset,
            start=None if start is None else start.tolist(),
            goal=None if goal is None else goal.tolist(),
            direct_blocked=None,
        )
        cases.append(case)
        if goal is None:
            attempts.append(dict(query=query, route=0, status="no_valid_endpoint_pair"))
            continue
        for route in range(routes):
            began = time.perf_counter()
            space = JointSpace(arm, cfg, boxes, start, seconds=2.5)
            planner_rng = np.random.default_rng(
                seed + family * 100003 + query * 101 + route
            )
            row = dict(query=query, route=route, status="rrt_failed")
            try:
                case["direct_blocked"] = not space.segment_free(start, goal)
                path = rrt_connect(
                    space, start, goal, planner_rng, step=0.18, max_iterations=500
                )
                if path is not None:
                    path = resample_joint(
                        shortcut(space, path, planner_rng, attempts=12)
                    )
                    space.deadline = time.monotonic() + 10
                    if not space.path_free(path):
                        raise ValueError("Teacher invalid after resampling")
                    from ..table_collision import path_free as table_path_free

                    if not table_path_free(arm, path, cfg):
                        raise ValueError(
                            "Teacher body intersects table between waypoints"
                        )
                    check_swept_obstacles(arm, path, boxes)
                    paths.append((query, path.astype(np.float32)))
                    row["status"] = "success"
            except ValueError as exc:
                row["reason"] = str(exc)
            row["elapsed_s"] = time.perf_counter() - began
            attempts.append(row)
    return cases, attempts, paths


def generate(base, folder, families=100, queries=6, routes=2, workers=4, seed=20261126):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "dataset.json").exists() or (folder / "dataset.npz").exists():
        raise ValueError("Choose a new dataset folder")
    if min(families, queries, routes, workers) < 1 or families < 10 or routes > 8:
        raise ValueError(
            "Need at least 10 families, positive sizes, and at most eight routes"
        )
    check_budget(folder, families * queries * routes * 16000 + 2_000_000)
    rows, attempts, paths, indices, splits = [], [], [], [], []
    args = [(base, f, queries, routes, seed) for f in range(families)]
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for family, (cases, trials, generated) in enumerate(
            pool.map(generate_family, args)
        ):
            offset = len(rows)
            rows.extend(cases)
            attempts.extend(dict(family=family, **t) for t in trials)
            for query, path in generated:
                paths.append(path)
                indices.append(offset + query)
                splits.append(cases[query]["split"])
            print(
                f"Families {family + 1}/{families}; valid teachers {len(paths)}/{len(attempts)}",
                flush=True,
            )
    arm = Arm(base["model_joint_limits_deg"], base["tool_offset_m"])
    report = dict(
        schema=2,
        task="joint_trajectory_3d",
        synthetic_only=True,
        hardware_executable=False,
        motor_io=False,
        seed=seed,
        scenarios=rows,
        attempts=attempts,
        limits_rad=[arm.lower.tolist(), arm.upper.tolist()],
        tool_offset_m=arm.tool_offset.tolist(),
        methods=list(METHODS),
        split_method="entire obstacle/table/workspace families held out, including all queries and route seeds",
        validation="runtime whole-arm object capsules, self collision, joint/workspace and continuous collision-body/table checks; 0.5-degree object sweeps",
        metrics=dict(
            families=families,
            scenarios=len(rows),
            valid_trajectories=len(paths),
            attempts=len(attempts),
            elapsed_s=time.perf_counter() - began,
            direct_blocked=sum(c["direct_blocked"] is True for c in rows),
        ),
        limitations=[
            "Geometric simulation, not contact/dynamics simulation.",
            "No camera noise, motor lag, friction, grasp contacts or success labels.",
            "Whole-arm capsules are approximate; boxes are axis aligned.",
            "Failed teachers retained; harder unsolved layouts are not silently labeled successful.",
        ],
    )
    np.savez_compressed(
        folder / "dataset.npz",
        paths=np.array(paths, np.float32).reshape(-1, 32, 5),
        scenario_index=np.array(indices, np.int32),
        split=np.array(splits),
    )
    report["dataset_sha256"] = hashlib.sha256(
        (folder / "dataset.npz").read_bytes()
    ).hexdigest()
    (folder / "dataset.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    check_budget(folder)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--folder", default="runs/flowreach/joint-trajectories-v1")
    parser.add_argument("--families", type=int, default=100)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--routes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--revalidate",
        help="Recheck an existing teacher dataset against collision-body table geometry",
    )
    args = parser.parse_args()
    report = (
        revalidate(args.revalidate, args.folder, args.workers)
        if args.revalidate
        else generate(
            load_config(args.config),
            args.folder,
            args.families,
            args.queries,
            args.routes,
            args.workers,
        )
    )
    print(json.dumps(report["metrics"]), flush=True)


def recheck_teacher(args):
    path, case, limits, tool = args
    from ..table_collision import path_free

    arm = Arm(np.rad2deg(limits), tool)
    return path_free(arm, path, case["config"])


def revalidate(source, destination, workers=4):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "dataset.json").exists():
        raise ValueError("Choose a new output folder")
    report = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    data = np.load(source / "dataset.npz")
    args = [
        (p, report["scenarios"][int(i)], report["limits_rad"], report["tool_offset_m"])
        for p, i in zip(data["paths"], data["scenario_index"])
    ]
    keep = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, valid in enumerate(pool.map(recheck_teacher, args, chunksize=8)):
            keep.append(valid)
            if (index + 1) % 200 == 0:
                print(
                    f"Collision-body recheck {index + 1}/{len(args)}; retained {sum(keep)}",
                    flush=True,
                )
    keep = np.asarray(keep, bool)
    from ..table_collision import minimum_clearance

    arm = Arm(np.rad2deg(report["limits_rad"]), report["tool_offset_m"])
    for case in report["scenarios"]:
        case["endpoint_body_valid"] = bool(
            case["goal"] is not None
            and minimum_clearance(arm, case["start"], case["config"]) >= 0
            and minimum_clearance(arm, case["goal"], case["config"]) >= 0
        )
    check_budget(
        destination,
        (source / "dataset.npz").stat().st_size
        + (source / "dataset.json").stat().st_size,
    )
    np.savez_compressed(
        destination / "dataset.npz", **{k: data[k][keep] for k in data.files}
    )
    report["parent_dataset_sha256"] = report["dataset_sha256"]
    report["dataset_sha256"] = hashlib.sha256(
        (destination / "dataset.npz").read_bytes()
    ).hexdigest()
    report["collision_model"] = (
        "URDF body convex hull / physical plane with adaptive continuous table checking; whole-arm object capsules"
    )
    report["metrics"].update(
        valid_trajectories=int(keep.sum()), body_table_rejected=int((~keep).sum())
    )
    report["body_table_rejected_teacher_indices"] = np.flatnonzero(~keep).tolist()
    (destination / "dataset.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    main()
