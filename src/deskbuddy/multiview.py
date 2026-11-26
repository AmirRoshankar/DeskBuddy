"""A4-anchored two-camera geometry; planar alignment is not elevated 3D calibration."""

import hashlib
import json
import time

import numpy as np

from .overlays import project
from .paper_calibration import fit_paper


def primary_signature(config):

    return hashlib.sha256(
        json.dumps(
            [
                config.get("T_base_camera"),
                config.get("test_bed"),
                config.get("assume_level_table", False),
                config.get("table_z_m"),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def blue_mask(rgb):

    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    return cv2.inRange(hsv, np.array([85, 65, 35]), np.array([135, 255, 255]))


def paper_candidate(rgb):

    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, np.array([0, 0, 110]), np.array([180, 95, 255]))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    candidates = []

    for contour in contours:
        hull = cv2.convexHull(contour)

        approx = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * 0.025, True)

        area = cv2.contourArea(approx) / (rgb.shape[0] * rgb.shape[1])

        if len(approx) == 4 and 0.03 < area < 0.75:
            p = approx[:, 0].astype(float)

            p = p[
                np.argsort(
                    np.arctan2(p[:, 1] - p[:, 1].mean(), p[:, 0] - p[:, 0].mean())
                )
            ]

            candidates.append((area, p))

    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def intrinsic_from_paper(h, size):
    """Estimate focal length assuming square pixels, centered principal point, no distortion.



    These assumptions are reported; a plane fit alone is not independent 3D validation.

    """

    width, height = size

    cx, cy = width / 2, height / 2

    a = h[:2, 0] - np.array([cx, cy]) * h[2, 0]

    b = h[:2, 1] - np.array([cx, cy]) * h[2, 1]

    lhs = np.array([a @ b, a @ a - b @ b])

    rhs = np.array([-h[2, 0] * h[2, 1], h[2, 1] ** 2 - h[2, 0] ** 2])

    inverse_squared = (lhs @ rhs) / max(lhs @ lhs, 1e-30)

    focal = np.sqrt(1 / inverse_squared) if inverse_squared > 0 else np.nan

    if not np.isfinite(focal) or not 0.3 * width < focal < 4 * width:
        return None

    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])


def blue_agreement(primary, secondary, pp, sp):

    import cv2

    rectangle = np.float32([[0, 0], [209, 0], [209, 296], [0, 296]])

    a = (
        cv2.warpPerspective(
            blue_mask(primary),
            cv2.getPerspectiveTransform(np.float32(pp), rectangle),
            (210, 297),
        )
        > 0
    )

    b = (
        cv2.warpPerspective(
            blue_mask(secondary),
            cv2.getPerspectiveTransform(np.float32(sp), rectangle),
            (210, 297),
        )
        > 0
    )

    kernel = np.ones((7, 7), np.uint8)

    da, db = (
        cv2.dilate(a.astype(np.uint8), kernel) > 0,
        cv2.dilate(b.astype(np.uint8), kernel) > 0,
    )

    if min(a.sum(), b.sum()) < 40:
        return None

    return float(((a & db).sum() / a.sum() + (b & da).sum() / b.sum()) / 2)


def calibrate_views(
    primary, secondary, pp, sp, intrinsics, config, depth, *, strict=True
):

    import cv2

    pp, sp = np.asarray(pp, float), np.asarray(sp, float)

    bed = fit_paper(
        pp,
        intrinsics,
        config["T_base_camera"],
        depth,
        strict=strict,
        table_z=config["table_z_m"] if config.get("assume_level_table") else None,
    )

    if (
        sp.shape != (4, 2)
        or not np.isfinite(sp).all()
        or not cv2.isContourConvex(sp.astype(np.float32))
        or abs(cv2.contourArea(sp.astype(np.float32))) < 500
    ):
        raise ValueError("Select the same four paper corners in both views, in order")

    obj = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])

    homography = cv2.getPerspectiveTransform(
        obj[:, :2].astype(np.float32), sp.astype(np.float32)
    )

    blue = blue_agreement(primary, secondary, pp, sp)

    warnings = list(bed.get("validation_warnings", []))

    if blue is not None and blue < 0.25 and strict:
        raise ValueError(
            "Blue markings do not align; correct the numbered corner correspondences"
        )

    if blue is not None and blue < 0.25:
        warnings.append("Blue markings do not align; review corner correspondence")

    k = intrinsic_from_paper(homography, (secondary.shape[1], secondary.shape[0]))

    result = dict(
        schema=1,
        validation_warnings=warnings,
        validation_passed=not warnings,
        calibrated_unix_s=time.time(),
        image_size=[secondary.shape[1], secondary.shape[0]],
        primary_pixels=pp.tolist(),
        secondary_pixels=sp.tolist(),
        H_secondary_paper=homography.tolist(),
        corners_base_m=bed["corners_base_m"],
        blue_agreement=blue,
        metric_3d_validated=False,
        intrinsics_assumptions="square pixels, centered principal point, zero lens distortion",
        primary_bed=bed,
    )

    if k is not None:
        ok, rvec, tvec = cv2.solvePnP(obj, sp, k, None, flags=cv2.SOLVEPNP_IPPE)

        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(obj, sp, k, None, rvec, tvec)

            base = np.asarray(bed["corners_base_m"])

            base_paper = np.eye(4)

            base_paper[:3, 0] = (base[1] - base[0]) / 0.210

            base_paper[:3, 1] = (base[3] - base[0]) / 0.297

            base_paper[:3, 2] = np.cross(base_paper[:3, 0], base_paper[:3, 1])

            base_paper[:3, 3] = base[0]

            camera_paper = np.eye(4)

            camera_paper[:3, :3] = cv2.Rodrigues(rvec)[0]

            camera_paper[:3, 3] = tvec.ravel()

            pose = base_paper @ np.linalg.inv(camera_paper)

            result.update(
                T_base_camera=pose.tolist(),
                intrinsics=dict(
                    fx=float(k[0, 0]),
                    fy=float(k[1, 1]),
                    ppx=float(k[0, 2]),
                    ppy=float(k[1, 2]),
                ),
            )

    if warnings:
        result["pose_validation"] = "Saved with warnings · elevated 3D unverified"

    else:
        validate_with_depth_features(
            result, primary, secondary, depth, intrinsics, config
        )

    result["primary_signature"] = primary_signature({**config, "test_bed": bed})

    return result


def validate_with_depth_features(result, primary, secondary, depth, intrinsics, config):
    """Check/refine elevated pose on held-out RGB feature matches anchored by RGB-D."""

    import cv2
    from scipy.optimize import least_squares

    sift = cv2.SIFT_create(nfeatures=1600)

    kp1, d1 = sift.detectAndCompute(cv2.cvtColor(primary, cv2.COLOR_RGB2GRAY), None)

    kp2, d2 = sift.detectAndCompute(cv2.cvtColor(secondary, cv2.COLOR_RGB2GRAY), None)

    if d1 is None or d2 is None:
        result["pose_validation"] = (
            "Not enough common RGB features; table-plane alignment only"
        )

        return

    matches = [
        a
        for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2)
        if a.distance < 0.68 * b.distance
    ]

    t = np.asarray(config["T_base_camera"])

    world, pixels = [], []

    for match in matches:
        u, v = kp1[match.queryIdx].pt

        x, y = int(round(u)), int(round(v))

        patch = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]

        valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 2)]

        if len(valid) < 5 or np.std(valid) > 0.012:
            continue

        def val(name):

            return (
                intrinsics[name]
                if isinstance(intrinsics, dict)
                else getattr(intrinsics, name)
            )

        z = float(np.median(valid))

        point = (
            t[:3, :3]
            @ np.array(
                [(u - val("ppx")) / val("fx") * z, (v - val("ppy")) / val("fy") * z, z]
            )
            + t[:3, 3]
        )

        if not (
            0.03 < point[0] < 0.55 and -0.4 < point[1] < 0.4 and -0.04 < point[2] < 0.45
        ):
            continue

        world.append(point)

        pixels.append(kp2[match.trainIdx].pt)

    result["depth_feature_matches"] = len(world)

    if len(world) < 20:
        result["pose_validation"] = (
            "Fewer than 20 RGB-D matches; table-plane alignment only"
        )

        return

    world, pixels = np.array(world), np.array(pixels)

    width, height = result["image_size"]

    k = np.array(
        [[width * 1.2, 0.0, width / 2], [0.0, width * 1.2, height / 2], [0.0, 0.0, 1.0]]
    )

    if result.get("intrinsics"):
        k[0, 0] = k[1, 1] = result["intrinsics"]["fx"]

    train = np.arange(len(world)) % 4 != 0

    ok, r, tr, inliers = cv2.solvePnPRansac(
        world[train], pixels[train], k, None, reprojectionError=6.0, iterationsCount=200
    )

    if not ok or inliers is None or len(inliers) < 12:
        result["pose_validation"] = (
            "Elevated feature fit inconclusive; table-plane alignment only"
        )

        return

    use = np.flatnonzero(train)[inliers.ravel()]

    obj = world[use]

    if np.ptp(obj[:, 2]) < 0.025:
        result["pose_validation"] = (
            "Common features are nearly planar; elevated pose unverified"
        )

        return

    base_corners = np.array(result["corners_base_m"])

    fit_points = np.vstack([obj, base_corners])

    fit_pixels = np.vstack([pixels[use], result["secondary_pixels"]])

    def residual(x):

        matrix = k.copy()

        matrix[0, 0] = matrix[1, 1] = np.exp(x[6])

        return (
            cv2.projectPoints(fit_points, x[:3], x[3:6], matrix, None)[0].reshape(-1, 2)
            - fit_pixels
        ).ravel()

    fit = least_squares(
        residual,
        np.r_[r.ravel(), tr.ravel(), np.log(k[0, 0])],
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=100,
        bounds=(
            np.r_[np.full(6, -20), np.log(width * 0.3)],
            np.r_[np.full(6, 20), np.log(width * 4)],
        ),
    )

    k[0, 0] = k[1, 1] = np.exp(fit.x[6])

    held = cv2.projectPoints(world[~train], fit.x[:3], fit.x[3:6], k, None)[0].reshape(
        -1, 2
    )

    errors = np.linalg.norm(held - pixels[~train], axis=1)

    result["heldout_reprojection_median_px"] = float(np.median(errors))

    corner_projection = cv2.projectPoints(base_corners, fit.x[:3], fit.x[3:6], k, None)[
        0
    ].reshape(-1, 2)

    corner_errors = np.linalg.norm(
        corner_projection - np.asarray(result["secondary_pixels"]), axis=1
    )

    result["refined_corner_errors_px"] = corner_errors.tolist()

    if np.median(errors) > 4 or np.quantile(errors, 0.8) > 8 or corner_errors.max() > 6:
        result["pose_validation"] = (
            "Held-out 3D check failed; table-plane alignment only"
        )

        return

    camera_base = np.eye(4)

    camera_base[:3, :3] = cv2.Rodrigues(fit.x[:3])[0]

    camera_base[:3, 3] = fit.x[3:6]

    pose = np.linalg.inv(camera_base)

    result.update(
        T_base_camera=pose.tolist(),
        metric_3d_validated=True,
        intrinsics=dict(
            fx=float(k[0, 0]), fy=float(k[1, 1]), ppx=width / 2, ppy=height / 2
        ),
        pose_validation="RGB-D fit and independent held-out elevated features passed",
    )


def calibration_current(config, size=None):

    calibration = config.get("secondary_camera_calibration")

    return bool(
        calibration
        and calibration.get("primary_signature") == primary_signature(config)
        and (size is None or list(size) == calibration.get("image_size"))
    )


def project_secondary(point, calibration, allow_planar=False):

    if calibration.get("metric_3d_validated") or (
        allow_planar and calibration.get("T_base_camera")
    ):
        return project(point, calibration["T_base_camera"], calibration["intrinsics"])

    return None


def secondary_surface_point(pixel, calibration, bounded=True):

    import cv2

    xy = cv2.perspectiveTransform(
        np.asarray(pixel, np.float64)[None, None],
        np.linalg.inv(calibration["H_secondary_paper"]),
    )[0, 0]

    if bounded and not (0 <= xy[0] <= 0.210 and 0 <= xy[1] <= 0.297):
        raise ValueError("Choose a point on the calibrated A4 sheet")

    p = np.asarray(calibration["corners_base_m"])

    return p[0] + (p[1] - p[0]) * xy[0] / 0.210 + (p[3] - p[0]) * xy[1] / 0.297
