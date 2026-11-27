"""Depth-anchored checks for manually paired elevated camera points."""

import numpy as np

from .overlays import project


def check_elevated_tip(result, observations, intrinsics, config):
    if not result.get("T_base_camera") or not result.get("intrinsics"):
        raise ValueError("Camera pose missing; check the A4 alignment first")

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    transform = np.asarray(
        (result.get("primary_camera_refinement") or {}).get(
            "T_base_camera", config["T_base_camera"]
        ),
        float,
    )
    points, errors, ids, rejected = [], [], [], []
    for number, observation in enumerate(observations, 1):
        u, v = observation["primary_pixel"]
        patch = observation.get("depth_patch")
        if patch is None:
            image = np.asarray(observation.get("depth"))
            x, y = int(round(u)), int(round(v))
            if image.ndim != 2 or not (
                2 <= x < image.shape[1] - 2 and 2 <= y < image.shape[0] - 2
            ):
                rejected.append(f"P{number}: no saved depth; recapture")
                continue
            patch = image[y - 2 : y + 3, x - 2 : x + 3]
        patch = np.asarray(patch, float)
        valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 2)]
        if len(valid) < 12:
            rejected.append(f"P{number}: depth {len(valid)}/25 pixels valid (need 12)")
            continue
        spread = float(np.ptp(valid))
        if spread > 0.025:
            rejected.append(
                f"P{number}: mixed foreground/background depth ({spread * 1000:.0f} mm spread; max 25)"
            )
            continue
        depth = float(np.median(valid))
        camera = np.array(
            [
                (u - value("ppx")) * depth / value("fx"),
                (v - value("ppy")) * depth / value("fy"),
                depth,
            ]
        )
        point = transform[:3, :3] @ camera + transform[:3, 3]
        height = float(point[2] - config.get("table_z_m", 0))
        if height < 0.025:
            rejected.append(
                f"P{number}: estimated height {height * 100:.1f} cm (need 2.5); check tip/depth or camera pose"
            )
            continue
        predicted = project(point, result["T_base_camera"], result["intrinsics"])
        if predicted is None:
            rejected.append(
                f"P{number}: projects behind webcam; camera pose inconsistent"
            )
            continue
        points.append(point)
        ids.append(number)
        errors.append(
            float(
                np.linalg.norm(np.asarray(predicted) - observation["secondary_pixel"])
            )
        )
    prefix = f"{len(observations)} pairs captured; {len(points)}/3 usable"
    if len(points) < 3:
        reasons = (
            "; ".join(rejected)
            if rejected
            else f"capture {3 - len(points)} more elevated pairs"
        )
        raise ValueError(prefix + " - " + reasons)
    coverage = float(np.linalg.norm(np.ptp(points, axis=0)))
    if coverage < 0.06:
        raise ValueError(
            prefix
            + f" - position spread {coverage * 100:.1f} cm; need 6 cm across poses"
        )
    passed = max(errors) <= 8
    worst = int(np.argmax(errors))
    status = (
        prefix
        + f" - {'PASS' if passed else 'FAIL'}: P{ids[worst]} camera mismatch {errors[worst]:.1f} px (max 8); pose spread {coverage * 100:.1f} cm"
    )
    if not passed:
        status += "; check matching tip pixels and camera alignment"
    if rejected:
        status += "; excluded " + "; ".join(rejected)
    return dict(
        passed=passed,
        error_px=errors,
        points_m=np.asarray(points).tolist(),
        point_ids=ids,
        rejected=rejected,
        status=status,
        threshold_px=8,
        depth_independent=False,
        method="held-out manually matched tip pixels and local RealSense depth",
    )
