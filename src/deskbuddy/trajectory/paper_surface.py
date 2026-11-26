"""Known calibrated A4 plane exclusion, without masking objects above it."""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..overlays import project
from ..test_bed import corners, table_height_at


def paper_surface(rgb, depth, intrinsics, config, robot_mask=None):
    shape = depth.shape
    empty = np.zeros(shape, bool)
    bed = corners(config, inset=False)
    if bed is None:
        return empty, empty, 0.0, dict(available=False)
    pixels = [project(p, config["T_base_camera"], intrinsics) for p in bed]
    if any(p is None for p in pixels):
        return empty, empty, 0.0, dict(available=False)
    image = Image.new("L", (shape[1], shape[0]))
    ImageDraw.Draw(image).polygon(pixels, fill=255)
    footprint = np.array(image, bool)
    # Calibration/rasterization uncertainty is confined to a small edge band.
    region = ndimage.binary_dilation(footprint, iterations=5)
    interior = ndimage.binary_erosion(footprint, iterations=5)
    edge_reference = interior & ~ndimage.binary_erosion(footprint, iterations=22)

    def value(k):
        return intrinsics[k] if isinstance(intrinsics, dict) else getattr(intrinsics, k)

    v, u = np.indices(shape)
    z = np.asarray(depth, float)
    camera = np.stack(
        ((u - value("ppx")) / value("fx") * z, (v - value("ppy")) / value("fy") * z, z),
        axis=-1,
    )
    transform = np.asarray(config["T_base_camera"])
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    # This uses the tilted calibrated plane, not one constant camera-depth value.
    height = base[..., 2] - table_height_at(base[..., :2], config)
    valid = np.isfinite(z) & (z > 0.1) & (z < 2)
    rgb = np.asarray(rgb, float)
    white = (rgb.min(axis=-1) > 90) & (np.ptp(rgb, axis=-1) < 65)
    anchors = edge_reference & valid & white & (np.abs(height) < 0.05)
    if robot_mask is not None:
        anchors &= ~robot_mask
    # Require observations spread around the sheet: a small white object cannot
    # redefine the paper's depth. Fit a bounded perception-only correction, never the robot calibration.
    rows, cols = np.nonzero(anchors)
    paper_rows, paper_cols = np.nonzero(footprint)
    spread = (
        len(rows) >= 200
        and len(paper_rows) > 0
        and np.ptp(rows) > 0.6 * np.ptp(paper_rows)
        and np.ptp(cols) > 0.6 * np.ptp(paper_cols)
    )
    bias, sigma = 0.0, 0.0
    correction = 0.0
    slopes = [0.0, 0.0]
    fit_accepted = False
    if spread:
        values = height[anchors]
        center = bed[:, :2].mean(axis=0)
        design = np.c_[base[anchors, :2] - center, np.ones(len(values))]
        keep = np.ones(len(values), bool)
        coefficients = np.zeros(3)
        for _ in range(5):
            coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
            residual = values - design @ coefficients
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            keep = np.abs(residual) <= max(0.004, min(0.012, 3 * scale))
            if keep.sum() < 200:
                break
        corner_offsets = np.c_[bed[:, :2] - center, np.ones(4)] @ coefficients
        if (
            keep.sum() >= 200
            and np.max(np.abs(corner_offsets)) <= 0.04
            and abs(coefficients[2]) <= 0.015
            and np.max(np.abs(coefficients[:2])) < 0.25
        ):
            correction = np.clip(
                (base[..., :2] - center) @ coefficients[:2] + coefficients[2],
                -0.04,
                0.04,
            )
            bias = float(coefficients[2])
            fit_accepted = True
            slopes = coefficients[:2].tolist()
            sigma = float(
                1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
            )
    band = float(np.clip(3 * sigma + 0.004, 0.008, 0.018))
    flat = region & valid & (np.abs(height - correction) <= band)
    return (
        flat,
        footprint,
        correction,
        dict(
            available=True,
            reference_pixels=int(anchors.sum()),
            reference_spread=bool(spread),
            fit_accepted=fit_accepted,
            depth_bias_m=bias,
            depth_plane_slopes=slopes,
            plane_noise_sigma_m=sigma,
            exclusion_band_m=band,
            excluded_pixels=int(flat.sum()),
        ),
    )
