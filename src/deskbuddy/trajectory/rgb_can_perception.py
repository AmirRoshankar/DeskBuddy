"""RGB-only can localization with calibrated views and a fixed cylinder prior.

No depth measurements enter this module. Dimensions are assumed (66 x 122 mm),
not measured. Fit projected silhouettes in the common calibrated world frame.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..test_bed import corners, table_height_at
from .perception import DetectedBox

RADIUS = 0.033
HEIGHT = 0.12222


def projector(transform, intrinsics, world_corners, image_corners):
    transform = np.asarray(transform)

    def val(name):
        return (
            intrinsics[name]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, name)
        )

    def raw(points):
        camera = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
        z = np.maximum(camera[:, 2], 1e-5)
        return np.c_[
            camera[:, 0] / z * val("fx") + val("ppx"),
            camera[:, 1] / z * val("fy") + val("ppy"),
        ]

    # Preserve the user-clicked paper alignment even when the estimated camera
    # pose has reprojection residuals. The elevated geometry remains estimated.
    correction = cv2.getPerspectiveTransform(
        raw(world_corners).astype(np.float32), np.float32(image_corners)
    )

    def project(points):
        return cv2.perspectiveTransform(
            raw(points).astype(np.float32)[None], correction
        )[0]

    return project


def cylinder(center, z):
    angles = np.arange(24) * (2 * np.pi / 24)
    ring = np.c_[np.cos(angles), np.sin(angles)] * RADIUS + center
    return np.vstack([np.c_[ring, np.full(24, z + dz)] for dz in (0, HEIGHT)])


def silhouette_roi(project, center, z):
    pixels = project(cylinder(center, z))
    return np.r_[pixels.min(0), pixels.max(0)]


def eligible(instances, paper_pixels, robot_mask=None):
    result = []
    for item in instances:
        mask = item["mask"]
        count = mask.sum()
        if count < 30:
            continue
        sheet = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(sheet, np.int32(paper_pixels), 1)
        # A sheet-size instance is not a can, even with a noisy depth boundary.
        intersection = (mask & (sheet > 0)).sum()
        if intersection / max(count + sheet.sum() - intersection, 1) > 0.5:
            continue
        # The encoder-projected arm mask is approximate and can overlap the
        # can near a grasp. It must not erase a strong, jointly consistent RGB
        # can silhouette merely because the calibration/mask is oversized.
        if (
            robot_mask is not None
            and (mask & robot_mask).sum() / count > 0.85
            and item["confidence"] < 0.6
        ):
            continue
        result.append(item)
    return result


def detect_rgb_cans(primary, secondary, config, calibration, robot_mask, intrinsics):
    report = dict(
        outlines=[],
        paper_surface=dict(available=False, method="RGB sheet exclusion"),
        secondary=dict(
            detections=[],
            fusion="RGB silhouettes + known can dimensions",
            geometry_estimated=True,
        ),
        geometry_prior=dict(diameter_m=2 * RADIUS, height_m=HEIGHT),
    )
    bed = config.get("test_bed")
    world = corners(config, inset=False)
    if bed is None or world is None or config.get("T_base_camera") is None:
        report["geometry_status"] = "Calibrate the table before RGB can localization"
        return [], report
    if config.get("secondary_camera") and calibration is None:
        report["geometry_status"] = "Awaiting fresh calibrated RGB second view"
        return [], report
    p = eligible(primary, bed["corner_pixels"], robot_mask)
    first = projector(config["T_base_camera"], intrinsics, world, bed["corner_pixels"])
    views = [first]
    if calibration is not None:
        if not calibration.get("T_base_camera") or not calibration.get("intrinsics"):
            report["geometry_status"] = (
                "Second camera pose required for RGB cylinder fitting"
            )
            return [], report
        second = projector(
            calibration["T_base_camera"],
            calibration["intrinsics"],
            world,
            calibration["secondary_pixels"],
        )
        views.append(second)
        s = eligible(secondary, calibration["secondary_pixels"])
    low, high = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    inverse = cv2.getPerspectiveTransform(
        np.float32(bed["corner_pixels"]), world[:, :2].astype(np.float32)
    )

    def fit_items(items, active_views, seed):
        rois = [np.asarray(item["roi"], float) for item in items]

        def residual(center):
            # Can mode assumes a level table; avoid rebuilding the footprint
            # thousands of times inside a two-variable numerical solve.
            z = table_z
            return np.concatenate(
                [
                    (silhouette_roi(view, center, z) - roi)
                    / max(np.linalg.norm(roi[2:] - roi[:2]), 30)
                    for view, roi in zip(active_views, rois)
                ]
            )

        fit = least_squares(
            residual,
            np.clip(seed, low[:2] + 1e-6, high[:2] - 1e-6),
            bounds=(low[:2], high[:2]),
            loss="soft_l1",
            f_scale=0.1,
            diff_step=0.001,
            max_nfev=25,
        )
        error = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
        return error, fit.x, items

    table_z = float(table_height_at(world[:1, :2], config)[0]) - bed.get(
        "clearance_margin_m", 0.003
    )

    # Fit each view once, associate in world XY, then jointly refine only plausible
    # pairs. Avoid an expensive Cartesian product of every background mask.
    def localize(items, view, homography):
        localized = []
        for item in items:
            roi = np.asarray(item["roi"])
            pixel = (roi[:2] + roi[2:]) / 2
            seed = cv2.perspectiveTransform(np.float32([[pixel]]), homography)[0, 0]
            fit = fit_items((item,), (view,), seed)
            if fit[0] < 0.20:
                localized.append(fit)
        return localized

    first_fits = localize(p, first, inverse)
    proposals = first_fits
    if calibration is not None:
        second_inverse = cv2.getPerspectiveTransform(
            np.float32(calibration["secondary_pixels"]), world[:, :2].astype(np.float32)
        )
        second_fits = localize(s, second, second_inverse)
        proposals = []
        for _, a, one in first_fits:
            for _, b, two in second_fits:
                if np.linalg.norm(a - b) < 0.07:
                    fit = fit_items(one + two, views, (a + b) / 2)
                    if fit[0] < 0.07:
                        proposals.append(fit)
    boxes, used = [], set()
    for error, center, items in sorted(proposals, key=lambda item: item[0]):
        if any(id(item) in used for item in items):
            continue
        if any(
            np.linalg.norm(center - (box.low[:2] + box.high[:2]) / 2) < RADIUS
            for box in boxes
        ):
            continue
        used.update(id(item) for item in items)
        z = table_z
        roi = tuple(items[0]["roi"])
        boxes.append(
            DetectedBox(
                np.r_[center - RADIUS, z],
                np.r_[center + RADIUS, z + HEIGHT],
                int(items[0]["mask"].sum()),
                HEIGHT,
                roi,
                source="YOLOE-26s RGB + can-size prior (no depth)",
                confidence=min(item["confidence"] for item in items),
            )
        )
        report["outlines"].append(
            dict(roi=roi, geometry="estimated", silhouette_fit_error=error)
        )
        if len(items) > 1:
            report["secondary"]["detections"].append(
                {k: v for k, v in items[1].items() if k != "mask"}
            )
    report["geometry_status"] = (
        "RGB cylinder fit; dimensions assumed, not depth-measured"
        if boxes
        else "No consistent can silhouette pair; check camera alignment if a can is visible"
    )
    return boxes, report
