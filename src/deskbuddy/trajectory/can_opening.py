"""Find an opened can's pour hole on the lid in the calibrated RGB view.

RGB only. The lid is a known circle (202 end, ~54 mm) at a known height above
the detected can base, so it projects to a known ellipse in the calibrated
primary camera. Inside that ellipse the opened pour hole is the darkest
compact region (the can interior is unlit), offset from the lid centre where
the tab rivet sits. Its pixel centroid is intersected with the lid plane to
give a base-frame XYZ.

This is an estimate: it inherits camera/A4 calibration error and assumes an
upright can of the nominal size. It returns a confidence and refuses when
the evidence is weak; it never commands hardware.
"""

import cv2
import numpy as np

from ..test_bed import corners
from .rgb_can_perception import projector

LID_PANEL_BELOW_SEAM_M = 0.006
# Typical stay-on-tab opening: about 20 x 14 mm, centred ~18 mm off-axis.
HOLE_AREA_M2 = np.pi * 0.010 * 0.007
HOLE_RADIUS_RANGE_M = (0.008, 0.025)


def lid_projector(config, intrinsics):
    bed = config.get("test_bed")
    world = corners(config, inset=False)
    if bed is None or world is None or config.get("T_base_camera") is None:
        raise ValueError("Calibrate the table and camera before finding the opening")
    return projector(config["T_base_camera"], intrinsics, world, bed["corner_pixels"])


def lid_plane_homography(project, center_xy, z):
    """Pixel -> (x, y) on the horizontal plane z near the can."""
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * 0.04
    world = square + center_xy
    pixels = project(np.c_[world, np.full(4, z)])
    return cv2.getPerspectiveTransform(np.float32(pixels), np.float32(world))


def fit_lid(gray, seam_px, search_px=30, scales=(0.9, 0.95, 1.0, 1.05, 1.1)):
    """Align the predicted seam ellipse to image edges (chamfer matching).

    Calibration and the RGB can localization put the predicted lid several
    pixels (about a centimetre) off; the seam is a strong edge, so a small
    translation/scale search recovers where the lid really is.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = gray.shape
    center = seam_px.mean(0)
    best = None
    for scale in scales:
        base = (seam_px - center) * scale + center
        for dx in range(-search_px, search_px + 1):
            for dy in range(-search_px, search_px + 1):
                points = np.int32(np.round(base + [dx, dy]))
                if (
                    (points < 0).any()
                    or (points[:, 0] >= width).any()
                    or (points[:, 1] >= height).any()
                ):
                    continue
                cost = float(np.minimum(distance[points[:, 1], points[:, 0]], 6).mean())
                if best is None or cost < best[0]:
                    best = (cost, scale, dx, dy)
    if best is None:
        raise ValueError("Can lid is not inside the camera view")
    cost, scale, dx, dy = best
    return dict(
        chamfer_px=cost, scale=scale, shift_px=[dx, dy], predicted_center_px=center
    )


MAX_VIEW_DISAGREEMENT_M = 0.02


def secondary_lid_center(image, config, can, radius, seam_z):
    """Lid centre from the second camera's edge fit, or None if unusable."""
    calibration = config.get("secondary_camera_calibration") or {}
    if not calibration.get("T_base_camera") or not calibration.get("intrinsics"):
        return None
    world = corners(config, inset=False)
    project = projector(
        calibration["T_base_camera"],
        calibration["intrinsics"],
        world,
        calibration["secondary_pixels"],
    )
    base = np.asarray(can["center_m"], float)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.c_[np.cos(angles), np.sin(angles)] * radius + base[:2]
    seam_px = project(np.c_[ring, np.full(len(ring), seam_z)])
    gray = cv2.cvtColor(np.asarray(image)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    try:
        fit = fit_lid(gray, seam_px, search_px=40)
    except ValueError:
        return None
    if fit["chamfer_px"] > 1.5:
        return None
    plane = lid_plane_homography(project, base[:2], seam_z)
    c0 = fit["predicted_center_px"]
    shift = (
        cv2.perspectiveTransform(np.float32([[c0 + fit["shift_px"]]]), plane)[0, 0]
        - cv2.perspectiveTransform(np.float32([[c0]]), plane)[0, 0]
    )
    return dict(center_xy=base[:2] + shift.astype(float), edge_fit_px=fit["chamfer_px"])


def detect_opening(rgb, intrinsics, config, can, debug=None, secondary=None):
    """Lid position (edge-fitted) and pour-hole estimate for ``can``.

    Returns a dict with ``lid_center_m`` (corrected can axis at the seam) and
    the hole. Raises ValueError when either cannot be established.
    """
    from ..can_discard import lid_radius

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Need an RGB image")
    project = lid_projector(config, intrinsics)
    base = np.asarray(can["center_m"], float)
    seam_z = base[2] + float(can["height_m"])
    panel_z = seam_z - LID_PANEL_BELOW_SEAM_M
    radius = lid_radius(can)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.c_[np.cos(angles), np.sin(angles)]
    gray = cv2.cvtColor(image[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    seam_px = project(np.c_[ring * radius + base[:2], np.full(len(ring), seam_z)])
    fit = fit_lid(gray, seam_px)
    if fit["chamfer_px"] > 1.5:
        raise ValueError(
            f"Lid outline not found in the image (edge fit {fit['chamfer_px']:.1f} px)"
        )
    c0, scale, shift = (
        fit["predicted_center_px"],
        fit["scale"],
        np.array(fit["shift_px"]),
    )

    def to_image(pixels):
        return (np.asarray(pixels) - c0) * scale + c0 + shift

    def from_image(pixels):
        return (np.asarray(pixels) - c0 - shift) / scale + c0

    seam_plane = lid_plane_homography(project, base[:2], seam_z)
    panel_plane = lid_plane_homography(project, base[:2], panel_z)

    def to_plane(pixel, homography):
        return cv2.perspectiveTransform(np.float32([[from_image(pixel)]]), homography)[
            0, 0
        ].astype(float)

    # The fitted ellipse is the predicted one moved in the image; undoing that
    # move and intersecting with the seam plane gives the corrected lid centre.
    fitted_center_px = to_image(project(np.r_[base[:2], seam_z][None])[0])
    predicted_xy = base[:2]
    shift_xy = (
        cv2.perspectiveTransform(np.float32([[c0 + shift]]), seam_plane)[0, 0]
        - cv2.perspectiveTransform(np.float32([[c0]]), seam_plane)[0, 0]
    ).astype(float)
    lid_xy = predicted_xy + shift_xy

    inner = to_image(
        project(np.c_[ring * (radius - 0.003) + base[:2], np.full(len(ring), panel_z)])
    )
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(inner))], 1)
    lid_pixels = int(mask.sum())
    if lid_pixels < 300:
        raise ValueError("Can lid is too small in the image to find the opening")
    expected = HOLE_AREA_M2 * lid_pixels / (np.pi * (radius - 0.003) ** 2)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    values = blurred[mask > 0].astype(np.uint8)
    median = float(np.median(values))
    otsu, _ = cv2.threshold(
        values[:, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = min(float(otsu), median * 0.6)
    dark = ((blurred <= threshold) & (mask > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if not expected * 0.2 <= area <= expected * 3.0:
            continue
        pixel = centroids[label]
        offset = to_plane(pixel, panel_plane) - predicted_xy
        off_axis = float(np.linalg.norm(offset))
        if (
            not HOLE_RADIUS_RANGE_M[0]
            <= off_axis
            <= min(HOLE_RADIUS_RANGE_M[1], radius - 0.002)
        ):
            continue
        darkness = float(blurred[labels == label].mean())
        contrast = (median - darkness) / max(median, 1.0)
        size_fit = float(np.exp(-abs(np.log(area / expected))))
        hole_xy = lid_xy + offset
        candidates.append(
            dict(
                pixel=[float(pixel[0]), float(pixel[1])],
                center_m=[float(hole_xy[0]), float(hole_xy[1]), float(panel_z)],
                azimuth_rad=float(np.arctan2(offset[1], offset[0])),
                off_axis_m=off_axis,
                area_px=int(area),
                expected_area_px=float(expected),
                contrast=float(contrast),
                confidence=float(np.clip(contrast, 0, 1) * size_fit),
            )
        )
    if debug is not None:
        debug.update(
            lid_outline_px=to_image(seam_px).tolist(),
            predicted_outline_px=seam_px.tolist(),
            search_px=inner.tolist(),
            threshold=threshold,
            lid_median=median,
            lid_fit=dict(fit, predicted_center_px=c0.tolist()),
            candidates=candidates,
        )
    # Fuse with the second camera's independent lid fit. Their disagreement
    # is mostly elevated-geometry calibration error; report it, and refuse
    # when the views are too far apart to trust either.
    views = dict(primary=lid_xy.tolist())
    if secondary is not None:
        other = secondary_lid_center(secondary, config, can, radius, seam_z)
        if other is not None:
            views["secondary"] = other["center_xy"].tolist()
            gap = float(np.linalg.norm(other["center_xy"] - lid_xy))
            views["disagreement_m"] = gap
            if gap <= MAX_VIEW_DISAGREEMENT_M:
                lid_xy = (lid_xy + other["center_xy"]) / 2
                shift_xy = lid_xy - predicted_xy
            else:
                # Large disagreement is camera calibration, not the can. Use
                # the RealSense view (the one that sees the hole); the hover
                # and in-hole camera checks then correct relative to the jaw.
                views["used"] = "RealSense only (views disagree)"
    lid = dict(
        lid_views_m=views,
        lid_center_m=[float(lid_xy[0]), float(lid_xy[1]), float(seam_z)],
        lid_correction_m=[float(shift_xy[0]), float(shift_xy[1])],
        lid_edge_fit_px=fit["chamfer_px"],
        lid_center_px=[float(v) for v in fitted_center_px],
    )
    if not candidates:
        raise ValueError("No pour opening found on the lid (is the can opened?)")
    best = max(candidates, key=lambda item: item["confidence"])
    # Hole offset is measured within the primary lid; re-anchor on the fused lid.
    angle = best["azimuth_rad"]
    best["center_m"][:2] = [
        float(lid_xy[0] + best["off_axis_m"] * np.cos(angle)),
        float(lid_xy[1] + best["off_axis_m"] * np.sin(angle)),
    ]
    if best["confidence"] < 0.15:
        raise ValueError(
            f"Pour opening uncertain (confidence {best['confidence']:.2f}); not used"
        )
    best.update(
        lid,
        source="primary RGB: edge-fitted lid ellipse + dark region on the panel",
        lid_radius_m=radius,
        estimated=True,
    )
    return best


def draw_opening(image, result, debug):
    """Annotated copy for logs/UI: lid outline, search region and the hole."""
    out = np.asarray(image)[..., :3].copy()
    if debug.get("lid_outline_px"):
        cv2.polylines(out, [np.int32(debug["lid_outline_px"])], True, (240, 200, 90), 1)
    if result is not None:
        u, v = map(int, np.round(result["pixel"]))
        cv2.circle(out, (u, v), 5, (255, 60, 60), 2)
    return out


def dark_spot_center(rgb, roi):
    """Centroid of a dark connected region within the upper 45% of a can box."""
    image = np.asarray(rgb)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = np.asarray(roi, float)
    x0, x1 = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
    bottom = min(height, int(np.ceil(y0 + 0.45 * (y1 - y0))))
    y0 = max(0, int(np.floor(y0)))
    if x1 - x0 < 6 or bottom - y0 < 6:
        raise ValueError("Webcam can top is outside the image or too small")
    gray = cv2.cvtColor(
        image[y0:bottom, x0:x1, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = min(float(otsu), median * 0.6, 85.0)
    dark = (gray <= threshold).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 5 or area > gray.size * 0.35:
            continue
        if x == 0 or y == 0 or x + w == gray.shape[1] or y + h == gray.shape[0]:
            continue
        contrast = (median - float(gray[labels == label].mean())) / max(median, 1.0)
        if contrast < 0.2:
            continue
        candidates.append((area * contrast, centers[label] + [x0, y0], contrast))
    if not candidates:
        raise ValueError("No distinct dark spot inside the webcam can top")
    _, pixel, contrast = max(candidates, key=lambda item: item[0])
    return dict(
        pixel=pixel.tolist(),
        confidence=float(np.clip(contrast, 0, 1)),
        search_roi_px=[x0, y0, x1, bottom],
        threshold=threshold,
    )


def detect_webcam_opening(rgb, config, can, debug=None):
    """Use only webcam RGB; camera calibration maps the centroid to the lid plane."""
    from ..can_discard import lid_radius
    from .rgb_can_perception import silhouette_roi

    calibration = config.get("secondary_camera_calibration") or {}
    if not calibration.get("T_base_camera") or not calibration.get("intrinsics"):
        raise ValueError("Align the webcam before mapping its hole position")
    world = corners(config, inset=False)
    if world is None:
        raise ValueError("Calibrate the tabletop first")
    project = projector(
        calibration["T_base_camera"],
        calibration["intrinsics"],
        world,
        calibration["secondary_pixels"],
    )
    base = np.asarray(can["center_m"], float)
    roi = silhouette_roi(project, base[:2], base[2])
    spot = dark_spot_center(rgb, roi)
    seam_z = float(base[2] + can["height_m"])
    panel_z = seam_z - LID_PANEL_BELOW_SEAM_M
    plane = lid_plane_homography(project, base[:2], panel_z)
    xy = cv2.perspectiveTransform(np.float32([[spot["pixel"]]]), plane)[0, 0]
    offset = xy - base[:2]
    spot.update(
        center_m=[float(xy[0]), float(xy[1]), panel_z],
        lid_center_m=[float(base[0]), float(base[1]), seam_z],
        lid_correction_m=[0.0, 0.0],
        lid_center_px=project(np.array([[*base[:2], seam_z]]))[0].tolist(),
        off_axis_m=float(np.linalg.norm(offset)),
        azimuth_rad=float(np.arctan2(offset[1], offset[0])),
        lid_radius_m=lid_radius(can),
        estimated=True,
        source="webcam RGB: dark spot in upper can box",
        camera="webcam",
        lid_views_m=dict(used="webcam only"),
    )
    if debug is not None:
        debug.update(search_roi_px=spot["search_roi_px"], threshold=spot["threshold"])
    return spot
