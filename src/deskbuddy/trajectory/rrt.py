"""Bidirectional RRT-Connect, checked shortcutting and corner-preserving resampling."""

import numpy as np


def rrt_connect(scene, start, goal, rng, step=0.015, max_iterations=1500):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if step <= 0 or max_iterations < 1:
        raise ValueError("Need a positive step and iteration budget")
    if not scene.segment_free(start, start) or not scene.segment_free(goal, goal):
        return None
    if scene.segment_free(start, goal):
        return np.array([start, goal])
    first, second = ([start], [-1]), ([goal], [-1])
    first_is_start = True

    def extend(tree, target):
        points, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(points) - target, axis=1)))
        delta = target - points[nearest]
        distance = np.linalg.norm(delta)
        if distance < 1e-12:
            return nearest, "reached"
        new = (
            target.copy()
            if distance <= step
            else points[nearest] + delta * step / distance
        )
        if not scene.segment_free(points[nearest], new):
            return nearest, "trapped"
        points.append(new)
        parents.append(nearest)
        return len(points) - 1, "reached" if distance <= step else "advanced"

    def branch(tree, index):
        points = []
        while index != -1:
            points.append(tree[0][index])
            index = tree[1][index]
        return points[::-1]

    for _ in range(max_iterations):
        sample = rng.uniform(scene.low, scene.high)
        index, status = extend(first, sample)
        if status != "trapped":
            target = first[0][index]
            while True:
                other, status = extend(second, target)
                if status != "advanced":
                    break
            if status == "reached":
                a, b = branch(first, index), branch(second, other)
                path = a + b[-2::-1] if first_is_start else b + a[-2::-1]
                return np.array(path)
        first, second = second, first
        first_is_start = not first_is_start
    return None


def shortcut(scene, path, rng, attempts=50):
    path = np.asarray(path, float).copy()
    if not scene.path_free(path):
        raise ValueError("Shortcut input is not collision-free")
    for _ in range(attempts):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j > i + 1 and scene.segment_free(path[i], path[j]):
            path = np.concatenate((path[: i + 1], path[j:]))
    return path


def resample(path, count=32):
    """Keep every corner so uniform sampling cannot cut through an obstacle."""
    path = np.asarray(path, float)
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or len(path) < 2
        or not np.isfinite(path).all()
    ):
        raise ValueError("Expected a finite 2D polyline")
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12]]
    if len(path) > count or count < 2 or len(path) < 2:
        raise ValueError("Not enough waypoint slots to preserve corners")
    length = np.linalg.norm(np.diff(path, axis=0), axis=1)
    intervals = np.ones(len(length), dtype=int)
    for _ in range(count - len(path)):
        intervals[np.argmax(length / intervals)] += 1
    points = [path[0]]
    for a, b, n in zip(path[:-1], path[1:], intervals):
        points.extend(
            a + (b - a) * fraction for fraction in np.linspace(0, 1, n + 1)[1:]
        )
    return np.array(points)
