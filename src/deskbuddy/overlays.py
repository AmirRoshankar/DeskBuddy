"""Project encoder/model points into the calibrated camera; not visual tracking."""

import numpy as np
from PIL import ImageDraw

from .geometry import transform_point
from .motion_overlay import draw_motion
from .test_bed import corners
from .workspace_geometry import box_edges, box_handles, target_guides


def project(point, transform, intrinsics):
    inverse = np.linalg.inv(np.asarray(transform, float))
    camera = transform_point(inverse, point)
    if camera[2] <= 0.03:
        return None

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    pixel = [
        value("fx") * camera[0] / camera[2] + value("ppx"),
        value("fy") * camera[1] / camera[2] + value("ppy"),
    ]
    return tuple(float(x) for x in pixel) if np.isfinite(pixel).all() else None


_VIEW_CACHE = {}


def view_project(point, config, intrinsics):
    """Project a base point into the RealSense image *as drawn for the user*.

    Anchored to the A4 corners the operator placed: table points land exactly
    on the clicked paper (the same homography used to turn clicks into table
    points), and elevated points get the same correction. Without A4 corner
    pixels it is the plain camera projection.
    """
    bed = config.get("test_bed") or {}
    pixels = bed.get("corner_pixels")
    if pixels is None or config.get("T_base_camera") is None:
        return project(point, config["T_base_camera"], intrinsics)
    key = repr(
        (
            config["T_base_camera"],
            pixels,
            bed.get("corners_base_m"),
            dict(
                (
                    k,
                    intrinsics[k]
                    if isinstance(intrinsics, dict)
                    else getattr(intrinsics, k),
                )
                for k in ("fx", "fy", "ppx", "ppy")
            ),
            config.get("assume_level_table"),
            config.get("table_z_m"),
        )
    )
    function = _VIEW_CACHE.get(key)
    if function is None:
        from .trajectory.rgb_can_perception import projector

        world = corners(config, inset=False)
        if world is None:
            return project(point, config["T_base_camera"], intrinsics)
        function = projector(config["T_base_camera"], intrinsics, world, pixels)
        if len(_VIEW_CACHE) > 8:
            _VIEW_CACHE.clear()
        _VIEW_CACHE[key] = function
    raw = project(point, config["T_base_camera"], intrinsics)
    if raw is None:
        return None
    pixel = function(np.asarray(point, float)[None])[0]
    return tuple(float(v) for v in pixel) if np.isfinite(pixel).all() else None


def visible_hover_bounds(surface, bounds, transform, intrinsics, size, margin=16):
    """Intersect a vertical target segment with the camera's visible frustum."""
    inverse = np.linalg.inv(np.asarray(transform, float))
    origin = transform_point(inverse, surface)
    direction = inverse[:3, 2]

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    fx, fy, cx, cy = (value(k) for k in ("fx", "fy", "ppx", "ppy"))
    width, height = size
    planes = (
        [fx, 0, cx - margin],
        [-fx, 0, width - 1 - margin - cx],
        [0, fy, cy - margin],
        [0, -fy, height - 1 - margin - cy],
    )
    low, high = bounds
    constraints = [(origin[2] - 0.031, direction[2])]
    constraints.extend((np.dot(p, origin), np.dot(p, direction)) for p in planes)
    for intercept, slope in constraints:
        if abs(slope) < 1e-12:
            if intercept < 0:
                raise ValueError("Hover target is outside the camera view")
        elif slope > 0:
            low = max(low, -intercept / slope)
        else:
            high = min(high, -intercept / slope)
    if low > high:
        raise ValueError("No visible hover height at this point")
    return float(low), float(high)


def grid_segments(config, intrinsics, divisions=4):
    bed = corners(config)
    if bed is not None and config.get("restrict_to_test_bed", True):
        result = []
        for s in np.linspace(0, 1, divisions + 1):
            for a, b in (
                (bed[0] * (1 - s) + bed[1] * s, bed[3] * (1 - s) + bed[2] * s),
                (bed[0] * (1 - s) + bed[3] * s, bed[1] * (1 - s) + bed[2] * s),
            ):
                p, q = (
                    view_project(a, config, intrinsics),
                    view_project(b, config, intrinsics),
                )
                if p is not None and q is not None:
                    result.append((p, q, s in (0, 1)))
        return result
    lo, hi = config["workspace_min_m"], config["workspace_max_m"]
    result = []
    for axis in (0, 1):
        for index, value in enumerate(np.linspace(lo[axis], hi[axis], divisions + 1)):
            a = [lo[0], lo[1], config["table_z_m"]]
            b = [hi[0], hi[1], config["table_z_m"]]
            a[axis] = b[axis] = value
            p = view_project(a, config, intrinsics)
            q = view_project(b, config, intrinsics)
            if p is not None and q is not None:
                result.append((p, q, index in (0, divisions)))
    return result


def estimated_goal_check(tip, target, config, intrinsics):
    gap = float(np.linalg.norm(np.asarray(tip) - target))
    tip_pixel = view_project(tip, config, intrinsics)
    target_pixel = view_project(target, config, intrinsics)
    pixel_gap = (
        float(np.linalg.norm(np.array(tip_pixel) - target_pixel))
        if tip_pixel and target_pixel
        else None
    )
    within = gap <= config.get("goal_tolerance_m", 0.01)
    overlap = pixel_gap is not None and pixel_gap <= 8.0
    return {
        "tip_base_m": np.asarray(tip).tolist(),
        "tip_pixel": tip_pixel,
        "target_pixel": target_pixel,
        "model_gap_mm": gap * 1000,
        "pixel_gap": pixel_gap,
        "model_within_tolerance": within,
        "projected_overlap": overlap,
        "estimated_goal_reached": bool(within and overlap),
        "independent_visual_measurement": False,
    }


def draw_overlays(
    image, arm, measured_q, target, surface, config, intrinsics, plan=None
):
    draw = ImageDraw.Draw(image)
    if config.get("show_workspace", True):
        for a, b in box_edges(config):
            p, q = (
                view_project(a, config, intrinsics),
                view_project(b, config, intrinsics),
            )
            if p is not None and q is not None:
                draw.line((p, q), fill="#77cbe8", width=2)
        for label, point, axis, _ in box_handles(config):
            pixel = view_project(point, config, intrinsics)
            if pixel is None:
                continue
            x, y = pixel
            if 8 <= x < image.width - 8 and 8 <= y < image.height - 8:
                draw.rectangle(
                    (x - 4, y - 4, x + 4, y + 4),
                    fill="#171719",
                    outline="#77cbe8",
                    width=2,
                )
                draw.text(
                    (min(x + 8, image.width - 50), max(42, y - 13)),
                    label,
                    fill="#b6eaff",
                    stroke_width=1,
                    stroke_fill="#171719",
                )
    for a, b, border in grid_segments(config, intrinsics):
        draw.line((a, b), fill="#57bfa9", width=2 if border else 1)
    lo, hi = config["workspace_min_m"], config["workspace_max_m"]
    bed = corners(config, inset=False)
    if bed is not None:
        outline = [view_project(p, config, intrinsics) for p in bed]
        if all(p is not None for p in outline):
            draw.line(outline + [outline[0]], fill="#ffe4a6", width=2)
        label = "A4 210 x 297 mm | grid inset 20 mm"
        if not config.get("restrict_to_test_bed", True):
            label = "PLAYGROUND BOX | A4 plane reference"
    else:
        label = f"SELECTION GRID {(hi[0] - lo[0]) * 100:.0f} x {(hi[1] - lo[1]) * 100:.0f} cm"
    draw_motion(
        image,
        arm,
        measured_q,
        plan,
        config,
        lambda p: view_project(p, config, intrinsics),
    )
    draw.text(
        (10, 10),
        label + f" | hover {config['hover_m'] * 100:.0f} cm",
        fill="#85ead0",
        stroke_width=1,
        stroke_fill="#102018",
    )

    def marker(pixel, color, radius, label):
        if pixel is None:
            return
        x, y = pixel
        if not (0 <= x < image.width and 0 <= y < image.height):
            return
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), outline=color, width=2
        )
        draw.line((x - radius - 3, y, x - 2, y), fill=color, width=1)
        draw.line((x + 2, y, x + radius + 3, y), fill=color, width=1)
        draw.text(
            (min(x + radius + 4, image.width - 120), max(36, y - radius - 12)),
            label,
            fill=color,
            stroke_width=1,
            stroke_fill="#101820",
        )

    if surface is not None:
        marker(
            view_project(surface, config, intrinsics),
            "#c0c4cd",
            4,
            "table click",
        )
    target_pixel = (
        view_project(target, config, intrinsics) if target is not None else None
    )
    if target is not None:
        from .motion_overlay import dashed

        for a, b in target_guides(target, config):
            p, q = (
                view_project(a, config, intrinsics),
                view_project(b, config, intrinsics),
            )
            if p is not None and q is not None:
                dashed(draw, p, q, "#68f1a6", width=2, dash=2)
                draw.ellipse((q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2), fill="#b6eaff")
    marker(target_pixel, "#68f1a6", 9, "tip goal (hover)")
    if measured_q is None:
        return None
    tip = arm.fk(measured_q)
    tip_pixel = view_project(tip, config, intrinsics)
    marker(tip_pixel, "#ff73dc", 6, "tip estimate")
    if target is None:
        return {
            "tip_base_m": tip.tolist(),
            "tip_pixel": tip_pixel,
            "independent_visual_measurement": False,
        }
    check = estimated_goal_check(tip, np.asarray(target), config, intrinsics)
    if plan is None and tip_pixel is not None and target_pixel is not None:
        draw.line((tip_pixel, target_pixel), fill="#f1b3df", width=1)
    pixels = f" / {check['pixel_gap']:.1f} px" if check["pixel_gap"] is not None else ""
    message = f"ESTIMATED GAP {check['model_gap_mm']:.1f} mm{pixels}"
    if check["estimated_goal_reached"]:
        message += " | within tolerance + projected overlap"
    draw.text((10, 27), message, fill="#f8c4ec", stroke_width=1, stroke_fill="#101820")
    return check


def draw_can_estimates(image, project_point, boxes, hole=None):
    """Estimated can cylinders (body, real lid edge) and the pour hole.

    ``project_point`` maps a base-frame point to a pixel (or None) for this
    view, so the same estimate is drawn consistently in every camera.
    """
    from .can_discard import LID_DIAMETER_M

    draw = ImageDraw.Draw(image)
    angles = np.linspace(0, 2 * np.pi, 33)
    ring = np.c_[np.cos(angles), np.sin(angles)]

    def polyline(points, color, width):
        pixels = [project_point(p) for p in points]
        for a, b in zip(pixels[:-1], pixels[1:]):
            if a is not None and b is not None:
                draw.line((a, b), fill=color, width=width)

    for box in boxes:
        low, high = np.asarray(box.low, float), np.asarray(box.high, float)
        size = high - low
        if not (0.08 < size[2] < 0.16 and max(size[:2]) < 0.09):
            continue
        center = (low[:2] + high[:2]) / 2
        radius = min(size[:2]) / 2
        bottom = np.c_[center + ring * radius, np.full(len(ring), low[2])]
        top = np.c_[center + ring * radius, np.full(len(ring), high[2])]
        lid = np.c_[center + ring * LID_DIAMETER_M / 2, np.full(len(ring), high[2])]
        polyline(bottom, "#77cbe8", 2)
        polyline(top, "#77cbe8", 1)
        polyline(lid, "#e8eef3", 2)
        for index in (0, 8, 16, 24):
            polyline([bottom[index], top[index]], "#77cbe8", 1)
    if hole is not None:
        point = np.asarray(hole, float)
        circle = np.c_[point[:2] + ring * 0.007, np.full(len(ring), point[2])]
        polyline(circle, "#ffd166", 3)
        for axis in (np.array([0.012, 0, 0]), np.array([0, 0.012, 0])):
            polyline([point - axis, point + axis], "#ffd166", 2)
        pixel = project_point(point)
        if pixel is not None:
            draw.text(
                (pixel[0] + 10, pixel[1] - 16),
                "Pour hole",
                fill="#ffd166",
                stroke_width=1,
                stroke_fill="#171719",
            )
