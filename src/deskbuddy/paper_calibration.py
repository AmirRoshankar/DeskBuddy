"""Known A4 rectangle pose from four explicit, ordered camera-image clicks."""

from copy import deepcopy

import numpy as np

from .test_bed import corners, table_height_at


def fit_paper(pixels, intrinsics, transform, depth=None, *, strict=True, table_z=None):
    import cv2

    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
        raise ValueError("Click four paper corners in order")
    polygon = pixels.astype(np.float32)
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 500:
        raise ValueError("Corners must surround the sheet in order, without crossing")

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    k = np.array(
        [[value("fx"), 0, value("ppx")], [0, value("fy"), value("ppy")], [0, 0, 1.0]]
    )
    obj = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])
    # Ordering defines which edge is 210 mm; operator reviews the projected outline.
    ok, rotation, translation = cv2.solvePnP(
        obj, pixels, k, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise ValueError("Paper pose could not be fitted")
    # IPPE is an initializer. Without refinement an exact frontal A4 fixture can
    # retain several-pixel error and invent centimetres of table tilt.
    rotation, translation = cv2.solvePnPRefineLM(
        obj, pixels, k, None, rotation, translation
    )
    camera = obj @ cv2.Rodrigues(rotation)[0].T + translation.ravel()
    projected = cv2.projectPoints(obj, rotation, translation, k, None)[0].reshape(-1, 2)
    if table_z is not None:
        from scipy.optimize import least_squares

        transform = np.asarray(transform, float)
        initial = camera @ transform[:3, :3].T + transform[:3, 3]
        inverse = np.linalg.inv(transform)

        def positions(parameters):
            x, y, angle = parameters
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            base = obj @ rotation.T + [x, y, float(table_z)]
            return base @ inverse[:3, :3].T + inverse[:3, 3]

        def residual(parameters):
            points = positions(parameters)
            uv = points @ k.T
            return (uv[:, :2] / uv[:, 2:] - pixels).ravel()

        yaw = np.arctan2(initial[1, 1] - initial[0, 1], initial[1, 0] - initial[0, 0])
        fit = least_squares(residual, [*initial[0, :2], yaw], loss="soft_l1", f_scale=2)
        camera = positions(fit.x)
        uv = camera @ k.T
        projected = uv[:, :2] / uv[:, 2:]
    error = np.linalg.norm(projected - pixels, axis=1)
    warnings = []
    if (camera[:, 2] <= 0.03).any():
        raise ValueError("Paper pose lies behind the camera")
    if error.max() > 4 and strict and table_z is None:
        raise ValueError(
            "Paper fit exceeds 4 px. Retry corners; first edge must be the short edge."
        )
    if error.max() > 4:
        warnings.append(f"A4 reprojection error {error.max():.1f} px exceeds 4 px")
    transform = np.asarray(transform, float)
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    depth_errors = []
    # Compare the plane at many interior pixels. Corner patches straddle the
    # sheet boundary and a single outlier must not veto a metric A4 pose.
    if depth is not None:
        for fx in np.linspace(0.15, 0.85, 9):
            for fy in np.linspace(0.15, 0.85, 11):
                point = (
                    camera[0]
                    + fx * (camera[1] - camera[0])
                    + fy * (camera[3] - camera[0])
                )
                uv = k @ point
                x, y = np.rint(uv[:2] / uv[2]).astype(int)
                if not (2 <= x < depth.shape[1] - 2 and 2 <= y < depth.shape[0] - 2):
                    continue
                patch = depth[y - 2 : y + 3, x - 2 : x + 3]
                valid = patch[np.isfinite(patch) & (patch > 0)]
                if (
                    valid.size >= 5
                    and np.quantile(valid, 0.8) - np.quantile(valid, 0.2) < 0.02
                ):
                    depth_errors.append(float(abs(np.median(valid) - point[2])))
        if len(depth_errors) >= 12 and np.median(depth_errors) > 0.04:
            warnings.append(
                f"Paper interior depth differs by {np.median(depth_errors) * 1000:.0f} mm median; A4 pose retained"
            )
    return dict(
        corners_base_m=base.tolist(),
        inset_m=0.02,
        dimensions_m=[0.210, 0.297],
        corner_pixels=pixels.tolist(),
        projected_pixels=projected.tolist(),
        reprojection_error_px=error.tolist(),
        depth_error_mm=(np.array(depth_errors) * 1000).tolist(),
        clearance_margin_m=0.003,
        validation_warnings=warnings,
        depth_check_samples=len(depth_errors),
    )


def apply_paper(config, bed):
    updated = deepcopy(config)
    updated["test_bed"] = bed
    inner = corners(updated)
    heights = table_height_at(inner[:, :2], updated)
    updated["workspace_min_m"][:2] = inner[:, :2].min(0).tolist()
    updated["workspace_max_m"][:2] = inner[:, :2].max(0).tolist()
    updated["table_z_m"] = float(heights.mean())
    updated["workspace_min_m"][2] = (
        float(heights.min()) + updated["link_clearance_m"] + 0.002
    )
    if updated["workspace_min_m"][2] >= updated["workspace_max_m"][2]:
        raise ValueError("Fitted paper lies outside the vertical workspace")
    # Remains unverified until supported hold and operator hand-clear confirmation.
    updated["calibration_verified"] = False
    return updated
