"""Estimate a near-horizontal tabletop from aligned depth, in the calibrated base frame."""

import numpy as np

from .geometry import validate_transform


def estimate_table(depth, intrinsics, transform):
    t = validate_transform(transform)
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    valid = (
        np.isfinite(depth)
        & (depth >= 0.30)
        & (depth <= 1.2)
        & (v > h * 0.50)
        & (v < h * 0.97)
        & (u > w * 0.05)
        & (u < w * 0.97)
    )
    z = depth[valid]
    camera = np.column_stack(
        (
            (u[valid] - intrinsics["ppx"]) * z / intrinsics["fx"],
            (v[valid] - intrinsics["ppy"]) * z / intrinsics["fy"],
            z,
        )
    )
    points = camera @ t[:3, :3].T + t[:3, 3]
    region = (
        (points[:, 0] > -0.15)
        & (points[:, 0] < 0.65)
        & (np.abs(points[:, 1]) < 0.45)
        & (points[:, 2] > -0.10)
        & (points[:, 2] < 0.14)
    )
    points = points[region]
    if len(points) < 500:
        raise ValueError("Not enough tabletop depth in the lower image")
    rng = np.random.default_rng(42)
    sample = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
    best = None
    for _ in range(220):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(normal[2]) < 0.96:
            continue
        offset = -normal @ a
        inliers = np.abs(sample @ normal + offset) < 0.004
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.mean() < 0.40:
        raise ValueError("No dominant near-horizontal table plane; inspect the view")
    surface = sample[best]
    center = surface.mean(0)
    _, _, vt = np.linalg.svd(surface - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(-normal @ center)
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1))))
    if tilt > 6:
        raise ValueError(
            f"Table plane tilts {tilt:.1f} degrees in the model frame; check the transform"
        )
    error = surface @ normal + offset
    return {
        "normal_base": normal.tolist(),
        "offset_m": offset,
        "height_at_base_origin_m": float(-offset / normal[2]),
        "tilt_deg": tilt,
        "rmse_mm": float(np.sqrt(np.mean(error**2)) * 1000),
        "inlier_fraction": float(best.mean()),
        "inliers": int(best.sum()),
    }


def conservative_height(plane, workspace_min, workspace_max):
    normal = np.asarray(plane["normal_base"])
    heights = [
        -(normal[0] * x + normal[1] * y + plane["offset_m"]) / normal[2]
        for x in (workspace_min[0], workspace_max[0])
        for y in (workspace_min[1], workspace_max[1])
    ]
    return float(max(heights) + 0.003)
