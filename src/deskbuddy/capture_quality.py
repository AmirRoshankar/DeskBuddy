"""Stricter depth checks for small gripper-tip calibration targets."""

import numpy as np

from .geometry import median_depth


def calibration_depth(frames, u, v):
    if len(frames) < 4:
        raise ValueError("Wait for fresh depth frames")
    h, w = frames[-1].shape
    if not (40 <= u < w - 40 and 40 <= v < h - 40):
        raise ValueError(
            "Keep the gripper tip away from the image edges (inside the inset box)"
        )
    values = []
    for depth in frames[-5:]:
        center = float(depth[v, u])
        if not 0.30 <= center <= 1.0:
            raise ValueError(
                "Tip depth is missing or outside 30–100 cm; use a pose about 35–65 cm from the camera"
            )
        distance = median_depth(depth, u, v, minimum=0.30, maximum=1.0)
        if abs(center - distance) > 0.006:
            raise ValueError(
                "The tip pixel and nearby depth disagree; adjust the viewing angle and click precisely"
            )
        values.append(distance)
    if np.ptp(values) > 0.006:
        raise ValueError("Tip depth is fluctuating; hold still or choose another pose")
    return float(np.median(values)), {
        "depth_samples_m": values,
        "depth_spread_mm": float(np.ptp(values) * 1000),
    }
