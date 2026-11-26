"""Camera-space handles for editing a robot-base-aligned box, without depth."""

from itertools import product

import numpy as np


def box_handles(config):
    low, high = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    center = (low + high) / 2
    result = [("Move XY", center, None, None)]
    for axis in range(3):
        for side, bound in enumerate((low, high)):
            point = center.copy()
            point[axis] = bound[axis]
            result.append(
                ("XYZ"[axis] + ("−" if side == 0 else "+"), point, axis, side)
            )
    return result


def box_edges(config):
    low, high = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    for bits in product((0, 1), repeat=3):
        for axis in range(3):
            if bits[axis] == 0:
                other = list(bits)
                other[axis] = 1
                yield np.where(bits, high, low), np.where(other, high, low)


def target_guides(target, config):
    """Three axis-aligned segments from the target to the nearest box faces."""
    target = np.asarray(target, float)
    low, high = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    if (target < low - 1e-9).any() or (target > high + 1e-9).any():
        return []
    guides = []
    for axis in range(3):
        end = target.copy()
        end[axis] = (
            low[axis]
            if target[axis] - low[axis] <= high[axis] - target[axis]
            else high[axis]
        )
        guides.append((target.copy(), end))
    return guides


def camera_ray(pixel, transform, intrinsics):
    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    ray = np.array(
        [
            (pixel[0] - value("ppx")) / value("fx"),
            (pixel[1] - value("ppy")) / value("fy"),
            1.0,
        ]
    )
    t = np.asarray(transform)
    direction = t[:3, :3] @ ray
    return t[:3, 3], direction / np.linalg.norm(direction)


def drag_box(config, handle, press_pixel, pixel, intrinsics):
    """Return a draft box from absolute drag positions; never accumulate deltas."""
    _, anchor, axis, side = handle
    transform = config["T_base_camera"]

    def coordinate(p):
        origin, ray = camera_ray(p, transform, intrinsics)
        if axis is None:
            if abs(ray[2]) < 1e-5:
                raise ValueError(
                    "View is parallel to XY; use the numeric bounds editor"
                )
            distance = (anchor[2] - origin[2]) / ray[2]
            if distance <= 0:
                raise ValueError("Bounds are behind the camera")
            return origin + ray * distance
        direction = np.eye(3)[axis]
        if 1 - (direction @ ray) ** 2 < 1e-4:
            raise ValueError(
                "Axis points into the camera; use the numeric bounds editor"
            )
        amount, distance = np.linalg.lstsq(
            np.column_stack((direction, -ray)), origin - anchor, rcond=None
        )[0]
        if distance <= 0:
            raise ValueError("Bounds are behind the camera")
        return amount

    delta = coordinate(pixel) - coordinate(press_pixel)
    low, high = (
        np.array(config["workspace_min_m"], float),
        np.array(config["workspace_max_m"], float),
    )
    if axis is None:
        delta[2] = 0
        low += delta
        high += delta
    elif side == 0:
        low[axis] = min(low[axis] + delta, high[axis] - 0.01)
    else:
        high[axis] = max(high[axis] + delta, low[axis] + 0.01)
    return dict(config, workspace_min_m=low.tolist(), workspace_max_m=high.tolist())
