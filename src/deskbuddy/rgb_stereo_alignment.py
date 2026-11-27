"""RGB-only stereo refinement anchored by the metric A4 sheet."""

import numpy as np


def refine_rgb_pairs(result, observations, intrinsics, config):
    import cv2
    from scipy.optimize import least_squares

    from .orange_calibration import triangulate_marker

    n = len(observations)
    if n < 6:
        raise ValueError(
            f"{n}/6 RGB pairs; need {6 - n} more (four to fit, two held out). Depth is not used"
        )
    if not result.get("T_base_camera") or not result.get("corners_base_m"):
        raise ValueError("Check A4 alignment first to establish camera poses and scale")

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    k1 = np.array(
        [[value("fx"), 0, value("ppx")], [0, value("fy"), value("ppy")], [0, 0, 1.0]]
    )
    primary = np.asarray(
        (result.get("primary_camera_refinement") or {}).get(
            "T_base_camera", config["T_base_camera"]
        ),
        float,
    )
    uv1 = np.array([p["primary_pixel"] for p in observations], float)
    uv2 = np.array([p["secondary_pixel"] for p in observations], float)
    if not np.isfinite(uv1).all() or not np.isfinite(uv2).all():
        raise ValueError("Nonfinite RGB click coordinates")
    rays = np.c_[uv1, np.ones(n)] @ np.linalg.inv(k1).T
    rays = rays @ primary[:3, :3].T
    train = n - 2
    if np.linalg.norm(np.ptp(uv1[:train], axis=0)) < 30:
        raise ValueError(
            "RGB pairs too close together; spread the first fitting poses across the view"
        )
    initial = np.linalg.inv(np.asarray(result["T_base_camera"], float))
    rotation = cv2.Rodrigues(initial[:3, :3])[0].ravel()
    ki = result["intrinsics"]
    width, height = result["image_size"]
    corners = np.asarray(result["corners_base_m"], float)
    corner_pixels = np.asarray(result["secondary_pixels"], float)

    def matrix(log_f):
        focal = np.exp(log_f)
        return np.array([[focal, 0, ki["ppx"]], [0, focal, ki["ppy"]], [0, 0, 1.0]])

    def project(points, x):
        return cv2.projectPoints(points, x[:3], x[3:6], matrix(x[6]), None)[0].reshape(
            -1, 2
        )

    def residual(x):
        points = primary[:3, 3] + rays[:train] * np.exp(x[7:])[:, None]
        return np.r_[
            (project(corners, x) - corner_pixels).ravel(),
            (project(points, x) - uv2[:train]).ravel(),
        ]

    start = np.r_[
        rotation, initial[:3, 3], np.log(ki["fx"]), np.full(train, np.log(0.5))
    ]
    low = np.r_[np.full(6, -np.inf), np.log(0.3 * width), np.full(train, np.log(0.08))]
    high = np.r_[np.full(6, np.inf), np.log(4 * width), np.full(train, np.log(3.0))]
    fit = least_squares(
        residual, start, bounds=(low, high), loss="soft_l1", f_scale=3, max_nfev=300
    )
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(fit.x[:3])[0]
    pose[:3, 3] = fit.x[3:6]
    pose = np.linalg.inv(pose)
    focal = float(np.exp(fit.x[6]))
    candidate = dict(
        T_base_camera=pose.tolist(),
        intrinsics=dict(fx=focal, fy=focal, ppx=ki["ppx"], ppy=ki["ppy"]),
    )
    first = dict(
        T_base_camera=primary.tolist(),
        intrinsics=dict(
            fx=value("fx"), fy=value("fy"), ppx=value("ppx"), ppy=value("ppy")
        ),
    )
    corner_error = float(
        np.max(np.linalg.norm(project(corners, fit.x) - corner_pixels, axis=1))
    )
    if not fit.success or corner_error > 6:
        raise ValueError(
            f"RGB pose fit failed: A4 corner error {corner_error:.1f} px (max 6); A4 corners and RGB pairs cannot fit one camera model; check corner order, matching tip feature, and whether camera moved"
        )
    checks = []
    for j in range(train, n):
        try:
            check = triangulate_marker(uv1[j], uv2[j], first, candidate)
        except ValueError as exc:
            raise ValueError(
                f"Held-out RGB P{j + 1}: {exc}; correct matching clicks or camera intrinsics"
            ) from exc
        point = np.asarray(check["point_m"])
        if point[2] - config.get("table_z_m", 0) < 0.025:
            raise ValueError(
                f"Held-out RGB P{j + 1}: triangulated height below 2.5 cm; use an elevated tip pose"
            )
        checks.append(check)
    error = max(max(p["reprojection_error_px"]) for p in checks)
    status = f"RGB stereo PASS: {train} fitting pairs + 2 held-out; check error {error:.1f} px; depth unused"
    return dict(
        passed=True,
        depth_independent=True,
        method="A4-anchored RGB bundle refinement with two held-out pairs",
        status=status,
        error_px=[max(p["reprojection_error_px"]) for p in checks],
        candidate=candidate,
        heldout=checks,
        corner_error_px=corner_error,
    )
