"""Incremental camera-aligned point cloud for a fixed external camera.

Missing observations preserve earlier surfaces. This is projective fusion, not
SLAM, object tracking, or a complete collision model. Camera movement invalidates it.
"""

import time

import numpy as np

from .geometry import deproject, transform_point
from .test_bed import surface_point


class SceneMap:
    def __init__(self):
        self.depth = None
        self.seen = None
        self.last_update = None
        self.variance = None
        self.pending_depth = None
        self.pending_count = None

    def update(self, depth, now=None):
        now = time.time() if now is None else now
        depth = np.asarray(depth, float)
        if self.depth is None or self.depth.shape != depth.shape:
            self.depth = np.full(depth.shape, np.nan)
            self.seen = np.full(depth.shape, -np.inf)
            self.variance = np.full(depth.shape, np.nan)
            self.pending_depth = np.full(depth.shape, np.nan)
            self.pending_count = np.zeros(depth.shape, dtype=np.uint8)
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        known = np.isfinite(self.depth)
        # Random-walk surface prior. Missing observations increase uncertainty.
        self.variance[known] += 0.001**2 * dt
        valid = np.isfinite(depth) & (depth >= 0.1) & (depth <= 2.0)
        # Assumed range-dependent measurement noise, to be tuned from observations.
        measurement_variance = (0.002 + 0.004 * np.where(valid, depth, 0.0) ** 2) ** 2
        new = valid & ~known
        self.depth[new] = depth[new]
        self.variance[new] = measurement_variance[new]
        self.seen[new] = now
        innovation = depth - self.depth
        gate = np.maximum(0.015, 4 * np.sqrt(self.variance + measurement_variance))
        outlier = valid & known & (np.abs(innovation) > gate)
        consistent = outlier & (np.abs(depth - self.pending_depth) < 0.01)
        self.pending_count[outlier & ~consistent] = 1
        self.pending_count[consistent] = np.minimum(
            self.pending_count[consistent] + 1, 3
        )
        self.pending_depth[outlier] = depth[outlier]
        changed = outlier & (self.pending_count >= 3)
        self.depth[changed] = depth[changed]
        self.variance[changed] = measurement_variance[changed]
        self.seen[changed] = now
        self.pending_count[changed] = 0
        accepted = valid & known & ~outlier
        gain = self.variance[accepted] / (
            self.variance[accepted] + measurement_variance[accepted]
        )
        self.depth[accepted] += gain * innovation[accepted]
        self.variance[accepted] *= 1 - gain
        self.seen[accepted] = now
        self.pending_count[accepted] = 0
        self.last_update = now

    def point(self, u, v, intrinsics, config, now=None):
        now = time.time() if now is None else now
        if config.get("test_bed"):
            return surface_point(u, v, intrinsics, config), {
                "source": "A4 plane estimate",
                "age_s": None,
            }
        if self.depth is None:
            raise ValueError("No scene observations yet")
        h, w = self.depth.shape
        if not 0 <= u < w or not 0 <= v < h:
            raise ValueError("Pixel outside image")
        x0, x1 = max(0, u - 8), min(w, u + 9)
        y0, y1 = max(0, v - 8), min(h, v + 9)
        patch = self.depth[y0:y1, x0:x1]
        valid = np.isfinite(patch)
        if not valid.any():
            raise ValueError(
                "This area has not been observed yet; map the tabletop or choose an observed area"
            )
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = (xx - u) ** 2 + (yy - v) ** 2
        index = np.argmin(np.where(valid, distance, np.inf))
        y, x = np.unravel_index(index, patch.shape)
        y += y0
        x += x0
        z = self.depth[y, x]

        def val(key):
            return (
                intrinsics[key]
                if isinstance(intrinsics, dict)
                else getattr(intrinsics, key)
            )

        camera = deproject(u, v, z, (val("fx"), val("fy"), val("ppx"), val("ppy")))
        age = max(0.0, now - self.seen[y, x])
        return transform_point(config["T_base_camera"], camera), {
            "source": "cached depth" if age > 0.2 else "observed depth",
            "age_s": float(age),
            "neighbor_pixels": float(np.sqrt(distance[y - y0, x - x0])),
            "depth_std_m": float(np.sqrt(self.variance[y, x])),
        }

    def point_cloud(self, intrinsics, transform, stride=4):
        if self.depth is None:
            return np.empty((0, 3))
        v, u = np.mgrid[
            0 : self.depth.shape[0] : stride, 0 : self.depth.shape[1] : stride
        ]
        z = self.depth[::stride, ::stride]
        valid = np.isfinite(z)

        def val(key):
            return (
                intrinsics[key]
                if isinstance(intrinsics, dict)
                else getattr(intrinsics, key)
            )

        points = np.column_stack(
            (
                (u[valid] - val("ppx")) * z[valid] / val("fx"),
                (v[valid] - val("ppy")) * z[valid] / val("fy"),
                z[valid],
            )
        )
        t = np.asarray(transform)
        return points @ t[:3, :3].T + t[:3, 3]

    def summary(self):
        return {
            "observed_fraction": float(np.isfinite(self.depth).mean())
            if self.depth is not None
            else 0.0,
            "last_update_unix_s": self.last_update,
            "method": "Kalman-filtered camera-aligned point cloud",
            "median_depth_std_m": float(np.nanmedian(np.sqrt(self.variance)))
            if self.variance is not None and np.isfinite(self.variance).any()
            else None,
        }
