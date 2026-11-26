"""Camera-fed model comparison without motor commands or execution entry points."""

import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from ..learned.data import check_budget
from .collision import Scene
from .evaluate import classical
from .learning import Dense, Field, condition, decode_path, sample
from .perception import grouped_footprints


def preview_scene(config, boxes):
    """Group extra detections conservatively into the two trained obstacle slots."""
    rectangles = grouped_footprints(boxes)
    return Scene(
        low=tuple(config["workspace_min_m"][:2]),
        high=tuple(config["workspace_max_m"][:2]),
        boxes=tuple(tuple(r) for r in rectangles),
        tool_radius=0.008,
        margin=0.010,
    )


class PlannerSuite:
    def __init__(self, folder="runs/flowreach/trajectories-v1/models-v3"):
        torch.set_num_threads(1)
        self.models = {}
        self.checkpoints = {}
        for name in ("mlp", "fm", "imf"):
            cp = torch.load(
                Path(folder) / (name + ".pt"), map_location="cpu", weights_only=True
            )
            if cp.get("task") != "cartesian_trajectory_2d":
                raise ValueError("Wrong trajectory checkpoint")
            model = (
                Dense(18, width=cp["width"])
                if name == "mlp"
                else Field(
                    cp["width"], auxiliary=cp.get("auxiliary_velocity_head", False)
                )
            )
            model.load_state_dict(cp["state_dict"])
            model.eval()
            self.models[name], self.checkpoints[name] = model, cp
        with torch.inference_mode():
            for name, model in self.models.items():
                sample(
                    model,
                    torch.zeros(1, 18),
                    torch.zeros(1, 60),
                    name,
                    8 if name == "fm" else 1,
                )

    def compare(
        self,
        scene,
        start,
        goal,
        candidates=1,
        seed=1729,
        allow_collision=False,
        methods=None,
    ):
        methods = ("mlp", "fm", "imf", "rrt") if methods is None else tuple(methods)
        if set(methods) - {"mlp", "fm", "imf", "rrt"}:
            raise ValueError("Unknown planner method")
        if candidates not in (1, 4):
            raise ValueError("Use one or four candidates")
        if len(scene.circles) + len(scene.boxes) > 2:
            raise ValueError("The trained models support up to two objects")
        start, goal = np.asarray(start), np.asarray(goal)
        if not allow_collision and (
            not scene.segment_free(start, start) or not scene.segment_free(goal, goal)
        ):
            raise ValueError(
                "Start or target overlaps estimated object clearance or bounds. Inspect 3D scene; object collisions can be overridden."
            )
        # Static XY path geometry is reversible; canonicalize to the training
        # left-to-right direction, then reverse output back before validation.
        reverse = start[0] > goal[0]
        a, b = (goal, start) if reverse else (start, goal)
        from dataclasses import asdict

        case = dict(index=seed, start=a.tolist(), goal=b.tolist(), scene=asdict(scene))
        noise = torch.randn(
            (candidates, 60), generator=torch.Generator().manual_seed(seed)
        )
        results = []
        with torch.inference_mode():
            for name in ("mlp", "fm", "imf"):
                if name not in methods:
                    continue
                count = 1 if name == "mlp" else candidates
                steps = 8 if name == "fm" else 1
                began = time.perf_counter()
                cp = self.checkpoints[name]
                n = cp["normalization"]
                c = (
                    torch.tensor(
                        (condition(case) - n["c_mean"]) / n["c_std"],
                        dtype=torch.float32,
                    )
                    .reshape(1, -1)
                    .repeat(count, 1)
                )
                prediction = sample(
                    self.models[name], c, noise[:count], name, steps
                ).numpy()
                paths = [
                    decode_path(p, case)
                    for p in prediction * np.array(n["y_std"]) + n["y_mean"]
                ]
                if reverse:
                    paths = [p[::-1].copy() for p in paths]
                generation_ms = (time.perf_counter() - began) * 1000
                began = time.perf_counter()
                valid = [p for p in paths if scene.path_free(p)]
                validation_ms = (time.perf_counter() - began) * 1000
                selected = (
                    min(
                        valid,
                        key=lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum(),
                    )
                    if valid
                    else None
                )
                results.append(
                    dict(
                        method=name,
                        nfe=steps,
                        candidates=count,
                        valid_candidates=len(valid),
                        raw_paths=[p.tolist() for p in paths],
                        selected_path=selected.tolist()
                        if selected is not None
                        else None,
                        generation_ms=generation_ms,
                        validation_ms=validation_ms,
                        total_ms=generation_ms + validation_ms,
                        fallback=False,
                        canonical_direction_reversed=bool(reverse),
                    )
                )
        if "rrt" not in methods:
            return results
        began = time.perf_counter()
        rrt_case = dict(index=seed, start=start.tolist(), goal=goal.tolist())
        path = classical(scene, rrt_case)
        elapsed = (time.perf_counter() - began) * 1000
        results.append(
            dict(
                method="rrt",
                nfe=0,
                candidates=1,
                valid_candidates=int(path is not None),
                raw_paths=[path.tolist()] if path is not None else [],
                selected_path=path.tolist() if path is not None else None,
                generation_ms=elapsed,
                validation_ms=0.0,
                total_ms=elapsed,
                fallback=False,
            )
        )
        return results


def save_preview(payload):
    root = Path("runs/flowreach/live-previews")
    root.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2) + "\n"
    check_budget(root, len(content.encode()))
    folder = root / str(uuid.uuid4())
    folder.mkdir()
    (folder / "preview.json").write_text(content, encoding="utf-8")
    return folder
