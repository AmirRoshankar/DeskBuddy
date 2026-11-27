"""RGB-only camera calibration from arm poses: no depth sensor involved.

During hand-guided recalibration the operator clicks the closed jaw tip in
the webcam image at 8+ arm poses. Encoder FK gives each tip's base-frame
position, so every click is a 2D-3D correspondence and the webcam pose
(plus its focal length, assuming square pixels and a centred principal
point) follows from PnP. The first N-2 points fit; the last two are held out
and their error is reported in millimetres at their depth.

The RealSense is then registered without depth too: the four A4 corners
clicked in the webcam are intersected with the level table to get their base
positions, and the RealSense pose follows from PnP on its own four corner
clicks with its factory intrinsics.
"""

import numpy as np

HELD_OUT = 2
MAX_ERROR_M = 0.012
PAPER = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])


def camera_matrix(focal, size):
    width, height = size
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def _pose(base, pixels, k):
    import cv2

    ok, rvec, tvec = cv2.solvePnP(base, pixels, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("Camera pose could not be solved")
    rvec, tvec = cv2.solvePnPRefineLM(base, pixels, k, None, rvec, tvec)
    projected = cv2.projectPoints(base, rvec, tvec, k, None)[0].reshape(-1, 2)
    camera_base = np.eye(4)
    camera_base[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_base[:3, 3] = tvec.ravel()
    return np.linalg.inv(camera_base), np.linalg.norm(projected - pixels, axis=1)


def metric_error(transform, k, base, pixels):
    """Pixel error converted to millimetres at each point's camera depth."""
    camera = (np.asarray(base) - transform[:3, 3]) @ transform[:3, :3]
    uv = camera @ k.T
    projected = uv[:, :2] / uv[:, 2:]
    pixel_error = np.linalg.norm(projected - pixels, axis=1)
    return pixel_error * camera[:, 2] / k[0, 0], pixel_error


def fit_camera(records, size, intrinsics=None):
    """Webcam (or any RGB camera) pose from clicked jaw tips and FK positions."""
    if len(records) < 8:
        raise ValueError("Need eight jaw-tip clicks: six to fit and two to check")
    pixels = np.array([r["pixel"] for r in records], float)
    base = np.array([r["base_point_model_m"] for r in records], float)
    if not np.isfinite(pixels).all() or not np.isfinite(base).all():
        raise ValueError("Calibration points must be finite")
    if ((pixels < 0) | (pixels >= np.asarray(size))).any():
        raise ValueError("Jaw-tip clicks must lie inside the camera image")
    # Duplicate observations overstate confidence and make focal/pose fitting
    # poorly conditioned even when the overall cloud happens to span 3D.
    distances = np.linalg.norm(base[:, None] - base[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if (distances.min(axis=1) < 0.003).any():
        raise ValueError(
            "Jaw-tip poses are too similar; spread captures throughout the workspace"
        )
    fit = len(records) - HELD_OUT
    spread = np.linalg.svd(base[:fit] - base[:fit].mean(0), compute_uv=False)
    if spread[-1] < 0.012 or np.linalg.norm(np.ptp(base[:fit], axis=0)) < 0.10:
        raise ValueError("Jaw-tip poses need more spread, including different heights")
    width = size[0]
    if intrinsics is not None:
        k = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["ppx"]],
                [0, intrinsics["fy"], intrinsics["ppy"]],
                [0, 0, 1.0],
            ]
        )
        candidates = [k]
    else:
        candidates = [
            camera_matrix(f, size) for f in np.geomspace(0.4 * width, 3 * width, 60)
        ]
    best = None
    for k in candidates:
        try:
            transform, residual = _pose(base[:fit], pixels[:fit], k)
        except ValueError:
            continue
        score = float(np.sqrt(np.mean(residual**2)))
        if best is None or score < best[0]:
            best = (score, k, transform)
    if best is None:
        raise ValueError("Camera pose could not be solved from the clicks")
    if intrinsics is None:
        # Refine the focal length around the best grid value.
        focal = best[1][0, 0]
        for f in np.linspace(focal * 0.93, focal * 1.07, 41):
            k = camera_matrix(f, size)
            transform, residual = _pose(base[:fit], pixels[:fit], k)
            score = float(np.sqrt(np.mean(residual**2)))
            if score < best[0]:
                best = (score, k, transform)
    _, k, transform = best
    train_mm, train_px = metric_error(transform, k, base[:fit], pixels[:fit])
    held_mm, held_px = metric_error(transform, k, base[fit:], pixels[fit:])
    camera_z = ((base - transform[:3, 3]) @ transform[:3, :3])[:, 2]
    if (camera_z <= 0.05).any():
        raise ValueError("Solved pose puts the arm behind the camera; recheck clicks")
    # Pixel residual translated at model-predicted depth is a lateral
    # reprojection-equivalent distance, not measured 3D tip accuracy.
    # Held-out poses must lie near the sampled volume; extrapolated checks can
    # otherwise appear reassuring while providing poor workspace coverage.
    train_min, train_max = base[:fit].min(axis=0), base[:fit].max(axis=0)
    check_outside = np.maximum(
        np.maximum(train_min - base[fit:], base[fit:] - train_max), 0
    )
    extrapolation = np.linalg.norm(check_outside, axis=1)
    coverage_ok = bool(np.max(extrapolation) <= 0.03)
    warnings = []
    if intrinsics is None:
        warnings.append(
            "Webcam focal length is estimated; lens distortion and principal point are not calibrated"
        )
    if not coverage_ok:
        warnings.append(
            "Check poses lie outside the sampled calibration volume; capture more varied poses"
        )
    warnings.append(
        "Errors are reprojection-equivalent estimates, not independently measured 3D tip accuracy"
    )
    return dict(
        T_base_camera=transform.tolist(),
        intrinsics=dict(
            fx=float(k[0, 0]), fy=float(k[1, 1]), ppx=float(k[0, 2]), ppy=float(k[1, 2])
        ),
        intrinsics_estimated=intrinsics is None,
        fit_points=fit,
        heldout_points=HELD_OUT,
        fit_rmse_mm=float(np.sqrt(np.mean(train_mm**2)) * 1000),
        fit_max_error_mm=float(train_mm.max() * 1000),
        fit_rmse_px=float(np.sqrt(np.mean(train_px**2))),
        heldout_error_mm=(held_mm * 1000).tolist(),
        heldout_error_px=held_px.tolist(),
        threshold_mm=MAX_ERROR_M * 1000,
        passed=bool(max(train_mm.max(), held_mm.max()) <= MAX_ERROR_M),
        heldout_extrapolation_mm=(extrapolation * 1000).tolist(),
        coverage_ok=coverage_ok,
        validation_warnings=warnings,
        error_interpretation="lateral reprojection-equivalent at model-predicted camera depth; not independent 3D accuracy",
        method="RGB PnP on clicked jaw tips; no depth",
    )


def table_points(pixels, transform, k, table_z):
    """Intersect webcam pixel rays with the level table plane."""
    transform = np.asarray(transform, float)
    out = []
    for u, v in np.asarray(pixels, float):
        ray = transform[:3, :3] @ np.linalg.solve(k, [u, v, 1.0])
        origin = transform[:3, 3]
        if abs(ray[2]) < 1e-9:
            raise ValueError("Paper corner ray is parallel to the table")
        t = (table_z - origin[2]) / ray[2]
        if t <= 0:
            raise ValueError("Paper corner lies behind the camera")
        out.append(origin + t * ray)
    return np.array(out)


def register_from_paper(webcam, web_pixels, rs_pixels, rs_intrinsics, size, table_z):
    """RealSense pose and the webcam calibration record, both from the A4.

    ``webcam`` is a passed fit_camera result. Returns (T_base_realsense,
    paper base corners, secondary calibration dict without its signature).
    """
    import cv2

    k_web = np.array(
        [
            [webcam["intrinsics"]["fx"], 0, webcam["intrinsics"]["ppx"]],
            [0, webcam["intrinsics"]["fy"], webcam["intrinsics"]["ppy"]],
            [0, 0, 1.0],
        ]
    )
    corners = table_points(web_pixels, webcam["T_base_camera"], k_web, table_z)
    sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
    if not (0.17 < sides[0] < 0.25 and 0.25 < sides[1] < 0.35):
        raise ValueError(
            f"Webcam paper edges measure {sides[0] * 1000:.0f} x {sides[1] * 1000:.0f} mm, "
            "not A4; click the short edge first and check the arm points"
        )
    k_rs = np.array(
        [
            [rs_intrinsics["fx"], 0, rs_intrinsics["ppx"]],
            [0, rs_intrinsics["fy"], rs_intrinsics["ppy"]],
            [0, 0, 1.0],
        ]
    )
    rs_pixels = np.asarray(rs_pixels, float)
    ok, rvec, tvec = cv2.solvePnP(
        corners, rs_pixels, k_rs, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise ValueError("RealSense pose could not be solved from the A4 corners")
    rvec, tvec = cv2.solvePnPRefineLM(corners, rs_pixels, k_rs, None, rvec, tvec)
    residual = np.linalg.norm(
        cv2.projectPoints(corners, rvec, tvec, k_rs, None)[0].reshape(-1, 2)
        - rs_pixels,
        axis=1,
    )
    camera_base = np.eye(4)
    camera_base[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_base[:3, 3] = tvec.ravel()
    realsense = np.linalg.inv(camera_base)
    homography = cv2.getPerspectiveTransform(
        PAPER[:, :2].astype(np.float32), np.float32(web_pixels)
    )
    secondary = dict(
        schema=1,
        validation_warnings=[]
        if residual.max() <= 4
        else [f"RealSense A4 corners {residual.max():.1f} px from the webcam paper"],
        validation_passed=bool(residual.max() <= 4),
        image_size=list(size),
        primary_pixels=rs_pixels.tolist(),
        secondary_pixels=np.asarray(web_pixels, float).tolist(),
        H_secondary_paper=homography.tolist(),
        corners_base_m=corners.tolist(),
        T_base_camera=webcam["T_base_camera"],
        intrinsics=webcam["intrinsics"],
        metric_3d_validated=bool(webcam["passed"]),
        intrinsics_assumptions="square pixels, centred principal point, no distortion"
        if webcam.get("intrinsics_estimated")
        else "provided",
        pose_validation=(
            f"webcam from {webcam['fit_points']} arm points, held-out "
            + ", ".join(f"{e:.1f}" for e in webcam["heldout_error_mm"])
            + " mm; RealSense from A4"
        ),
        hand_eye=webcam,
        operator_accepted_estimate=bool(
            webcam.get("operator_accepted_estimate", False)
        ),
        realsense_corner_residual_px=residual.tolist(),
    )
    return realsense, corners, secondary
