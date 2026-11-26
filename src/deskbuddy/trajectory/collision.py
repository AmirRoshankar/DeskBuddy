"""Continuous 2D segment checks for a circular tool and inflated obstacles."""

from dataclasses import dataclass

import numpy as np


def point_segment_distance(point, a, b):
    delta = b - a
    fraction = np.clip((point - a) @ delta / max(delta @ delta, 1e-30), 0, 1)
    return float(np.linalg.norm(point - (a + fraction * delta)))


def intersects_box(a, b, low, high):
    enter, leave = 0.0, 1.0
    for axis in range(2):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-14:
            if a[axis] < low[axis] or a[axis] > high[axis]:
                return False
        else:
            near, far = sorted(
                ((low[axis] - a[axis]) / delta, (high[axis] - a[axis]) / delta)
            )
            enter, leave = max(enter, near), min(leave, far)
            if enter > leave:
                return False
    return True


@dataclass
class Scene:
    low: tuple = (0.0, 0.0)
    high: tuple = (0.21, 0.297)
    circles: tuple = ()  # (cx, cy, radius)
    boxes: tuple = ()  # (xmin, ymin, xmax, ymax)
    tool_radius: float = 0.008
    margin: float = 0.004

    def __post_init__(self):
        low, high = np.asarray(self.low), np.asarray(self.high)
        if (
            low.shape != (2,)
            or high.shape != (2,)
            or not np.isfinite([low, high]).all()
            or (high <= low).any()
        ):
            raise ValueError("Invalid 2D workspace")
        if (
            not np.isfinite([self.tool_radius, self.margin]).all()
            or min(self.tool_radius, self.margin) < 0
        ):
            raise ValueError("Invalid inflation")
        for circle in self.circles:
            if len(circle) != 3 or not np.isfinite(circle).all() or circle[2] <= 0:
                raise ValueError("Invalid circle")
        for box in self.boxes:
            if (
                len(box) != 4
                or not np.isfinite(box).all()
                or (np.array(box[2:]) <= box[:2]).any()
            ):
                raise ValueError("Invalid rectangle")

    @property
    def inflation(self):
        return self.tool_radius + self.margin

    def segment_free(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            return False
        if (np.minimum(a, b) < self.low).any() or (np.maximum(a, b) > self.high).any():
            return False
        for cx, cy, radius in self.circles:
            if (
                point_segment_distance(np.array([cx, cy]), a, b)
                <= radius + self.inflation
            ):
                return False
        for box in self.boxes:
            # Axis-wise inflation is conservative at rectangle corners.
            if intersects_box(
                a,
                b,
                np.array(box[:2]) - self.inflation,
                np.array(box[2:]) + self.inflation,
            ):
                return False
        return True

    def path_free(self, path):
        path = np.asarray(path, float)
        return (
            path.ndim == 2
            and path.shape[1] == 2
            and len(path) >= 2
            and all(self.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))
        )

    def clearance(self, path):
        """Minimum distance to inflated geometry, for already validated paths."""
        result = float("inf")
        for a, b in zip(np.asarray(path)[:-1], np.asarray(path)[1:]):
            for cx, cy, radius in self.circles:
                result = min(
                    result,
                    point_segment_distance(np.array([cx, cy]), a, b)
                    - radius
                    - self.inflation,
                )
            for box in self.boxes:
                low, high = (
                    np.array(box[:2]) - self.inflation,
                    np.array(box[2:]) + self.inflation,
                )
                if intersects_box(a, b, low, high):
                    return 0.0
                endpoint_gap = min(
                    np.linalg.norm(p - np.clip(p, low, high)) for p in (a, b)
                )
                corner_gap = min(
                    point_segment_distance(np.array([x, y]), a, b)
                    for x in (low[0], high[0])
                    for y in (low[1], high[1])
                )
                result = min(result, float(endpoint_gap), corner_gap)
        return result
