"""Camera optical coordinates: x right, y down, z forward. All lengths in metres."""

import numpy as np


def validate_transform(transform):
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Transform must be a finite 4x4 matrix")
    r = t[:3, :3]
    if not (
        np.allclose(t[3], [0, 0, 0, 1])
        and np.allclose(r.T @ r, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(r), 1, atol=1e-6)
    ):
        raise ValueError("Transform must contain a proper rigid rotation")
    return t


def transform_point(transform, point):
    t = validate_transform(transform)
    p = np.asarray(point, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("Point must be a finite XYZ vector")
    return t[:3, :3] @ p + t[:3, 3]


def deproject(u, v, depth_m, intrinsics):
    """Pinhole projection; live camera uses the SDK to account for distortion."""
    fx, fy, cx, cy = intrinsics
    if (
        not np.isfinite([u, v, depth_m, fx, fy, cx, cy]).all()
        or min(fx, fy, depth_m) <= 0
    ):
        raise ValueError("Invalid pixel, depth, or intrinsics")
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def median_depth(depth_m, u, v, radius=2, minimum=0.1, maximum=2.0):
    """Reject missing depth; median reduces isolated stereo outliers."""
    h, w = depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Pixel outside image")
    patch = depth_m[
        max(0, v - radius) : min(h, v + radius + 1),
        max(0, u - radius) : min(w, u + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch >= minimum) & (patch <= maximum)]
    if valid.size < 5:
        raise ValueError("Insufficient valid depth near click")
    if np.ptp(np.percentile(valid, [10, 90])) > 0.02:
        raise ValueError("Depth discontinuity: click away from object edges")
    return float(np.median(valid))


def fit_rigid(camera_points, base_points):
    """Kabsch fit: p_base = R @ p_camera + t. Planar, noncollinear points work."""
    a, b = np.asarray(camera_points, float), np.asarray(base_points, float)
    if a.ndim != 2 or a.shape[1] != 3 or a.shape != b.shape or len(a) < 4:
        raise ValueError("Need at least four matching 3D points")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Calibration points must be finite")
    ac, bc = a - a.mean(0), b - b.mean(0)
    if min(np.linalg.matrix_rank(ac), np.linalg.matrix_rank(bc)) < 2:
        raise ValueError("Calibration points must not be collinear")
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    r = vt.T @ correction @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, b.mean(0) - r @ a.mean(0)
    errors = np.linalg.norm(a @ r.T + t[:3, 3] - b, axis=1)
    return validate_transform(t), errors
