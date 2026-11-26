"""Local elastic-band refinement of learned proposals; never calls RRT."""

import numpy as np
from scipy.optimize import minimize


def refine_route(path, scene, max_shift=0.065):
    raw = np.asarray(path, float)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid learned route")
    if scene.path_free(raw):
        return raw, dict(applied=False, max_shift_m=0.0)
    if not scene.segment_free(raw[0], raw[0]) or not scene.segment_free(
        raw[-1], raw[-1]
    ):
        raise ValueError(
            "Start or target overlaps estimated object clearance. Inspect 3D scene or turn on Allow predicted collisions."
        )
    if scene.circles:
        raise ValueError("Live refinement currently requires rectangular objects")
    if not scene.boxes:
        refined = np.clip(raw, scene.low, scene.high)
        displacement = float(np.linalg.norm(refined - raw, axis=1).max())
        if displacement > max_shift or not scene.path_free(refined):
            raise ValueError("Local route refinement exceeds its displacement budget")
        return refined, dict(
            applied=True,
            method="workspace projection",
            iterations=0,
            max_shift_m=displacement,
        )
    if len(raw) < 3:
        raise ValueError("Local route refinement needs an interior waypoint")
    boxes = np.array(scene.boxes)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    halves = (boxes[:, 2:] - boxes[:, :2]) / 2 + scene.inflation + 0.002
    low, high = np.array(scene.low), np.array(scene.high)

    def expand(x):
        return np.vstack((raw[0], x.reshape(-1, 2), raw[-1]))

    def clearance(x):
        points = expand(x)
        # A fine set of interpolated constraints guides the optimizer; exact
        # continuous segment checks remain the acceptance boundary below.
        samples = (
            points[:-1, None, :] * (1 - np.linspace(0, 1, 9)[None, :, None])
            + points[1:, None, :] * np.linspace(0, 1, 9)[None, :, None]
        ).reshape(-1, 2)
        delta = np.abs(samples[:, None, :] - centers) - halves
        signed = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(
            delta.max(axis=-1), 0
        )
        return signed.ravel()

    def cost(x):
        points = expand(x)
        return float(
            np.sum((points - raw) ** 2)
            + 0.15 * np.sum(np.diff(points, n=2, axis=0) ** 2)
        )

    def displacement_budget(x):
        return max_shift**2 - np.sum((expand(x)[1:-1] - raw[1:-1]) ** 2, axis=1)

    bounds = [
        (max(low[j], p[j] - max_shift), min(high[j], p[j] + max_shift))
        for p in raw[1:-1]
        for j in range(2)
    ]
    if any(a >= b for a, b in bounds):
        raise ValueError("Learned route is too far outside the workspace to refine")
    initial = np.clip(raw[1:-1], low, high).ravel()
    result = minimize(
        cost,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            dict(type="ineq", fun=clearance),
            dict(type="ineq", fun=displacement_budget),
        ],
        options=dict(maxiter=80, ftol=1e-9),
    )
    refined = expand(result.x)
    displacement = np.linalg.norm(refined - raw, axis=1).max()
    if (
        not result.success
        or displacement > max_shift + 1e-5
        or not scene.path_free(refined)
    ):
        raise ValueError("Local route refinement could not provide gripper clearance")
    return refined, dict(
        applied=True,
        method="local elastic-band",
        iterations=int(result.nit),
        max_shift_m=float(np.linalg.norm(refined - raw, axis=1).max()),
    )
