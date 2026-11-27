"""Offline-trained iMF box-grasp proposals; deliberately has no robot interface."""

from pathlib import Path

import numpy as np


def propose(
    size_m,
    yaw_rad=0.0,
    candidates=16,
    seed=1739,
    checkpoint="runs/deskbuddy/box-grasps-v1/imf.pt",
):
    import torch

    from ..trajectory.learning import Field, sample
    from .grasp_lab import geometric_metrics, unit

    size = np.asarray(size_m, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Enter positive finite box dimensions")
    cp = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if (
        cp.get("task") not in ("synthetic_box_grasp", "synthetic_can_grasp")
        or cp.get("method") != "imf"
    ):
        raise ValueError("Expected the synthetic-box iMF grasp checkpoint")
    model = Field(cp["width"], condition_dim=5, state_dim=8)
    model.load_state_dict(cp["state_dict"])
    model.eval()
    if cp.get("profile") == "can_rim":
        angles = torch.linspace(-torch.pi, torch.pi, candidates + 1)[:-1] + yaw_rad
        c = torch.tensor([*(size / 0.05), 1.0, 0.0], dtype=torch.float32).repeat(
            candidates, 1
        )
        c[:, 3], c[:, 4] = angles.cos(), angles.sin()
    else:
        c = torch.tensor(
            [[*(size / 0.05), np.cos(2 * yaw_rad), np.sin(2 * yaw_rad)]],
            dtype=torch.float32,
        ).repeat(candidates, 1)
    noise = torch.randn((candidates, 8), generator=torch.Generator().manual_seed(seed))
    with torch.inference_mode():
        output = sample(model, c, noise, "imf", 1)
        valid = geometric_metrics(output, c, True, profile=cp.get("profile", "box"))
        output[:, 3:7] = unit(output[:, 3:7])
    results = []
    for index, (row, ok) in enumerate(zip(output.numpy(), valid)):
        results.append(
            dict(
                index=index,
                grasp_mode="rim" if cp.get("profile") == "can_rim" else "body",
                offset_m=(row[:3] * 0.05).tolist(),
                quaternion_wxyz=row[3:7].tolist(),
                opening_m=float(row[7] * 0.05),
                geometric_valid=bool(ok),
            )
        )
    return sorted(results, key=lambda item: not item["geometric_valid"])


def scene_target(snapshot):
    """Prefer confirmed keys, then a selected nearby shared object; no size entry."""
    from ..grasp_keys import detect_key_target

    report = snapshot.get("report") or {}
    if snapshot.get("intrinsics") is not None:
        try:
            point, key = detect_key_target(
                report, snapshot["intrinsics"], snapshot["config"]
            )
            from ..test_bed import surface_point

            x1, y1, x2, y2 = key["pixel_roi"]
            corners = np.array(
                [
                    surface_point(x, y, snapshot["intrinsics"], snapshot["config"])
                    for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                ]
            )
            size = np.r_[np.maximum(np.ptp(corners[:, :2], axis=0), 0.004), 0.003]
            return dict(
                center_m=np.asarray(point).tolist(),
                size_m=size.tolist(),
                source="camera keys · 3 mm thickness assumption",
                yaw_rad=0.0,
            )
        except ValueError:
            pass
    boxes = snapshot.get("boxes", [])
    if not boxes:
        raise ValueError(
            "No confirmed scene object or keys. Keep object visible in both cameras."
        )
    selected = snapshot.get("selected")

    def cost(box):
        return (
            np.linalg.norm((box.low[:2] + box.high[:2]) / 2 - np.asarray(selected)[:2])
            if selected is not None
            else -(box.confidence or 0)
        )

    box = min(boxes, key=cost)
    center = (box.low + box.high) / 2
    center[2] = box.low[2]
    return dict(
        center_m=center.tolist(),
        size_m=np.maximum(box.high - box.low, 0.001).tolist(),
        source="shared two-camera object box",
        yaw_rad=0.0,
    )


def propose_scene(snapshot):
    target = scene_target(snapshot)
    rows = propose(target["size_m"], target["yaw_rad"], candidates=32)
    # Rank valid geometry first, then center error and opening clearance; this is
    # not a learned physical success probability or an IK-reachability guarantee.
    for row in rows:
        row["score"] = float(
            np.linalg.norm(row["offset_m"][:2]) / 0.003
            + abs(row["opening_m"] - min(target["size_m"][:2]) - 0.006) / 0.01
        )
    rows.sort(key=lambda row: (not row["geometric_valid"], row["score"]))
    target["candidates"] = rows
    target["best"] = next((row for row in rows if row["geometric_valid"]), None)
    size = np.array(target["size_m"])
    target["outside_training_range"] = bool(
        (size < [0.012, 0.012, 0.008]).any() or (size > [0.04, 0.04, 0.035]).any()
    )
    return target
