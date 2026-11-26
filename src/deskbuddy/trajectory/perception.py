"""Depth-above-table object boxes for live previews; not a certified collision map."""

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..test_bed import corners, table_height_at


@dataclass
class DetectedBox:
    low: np.ndarray
    high: np.ndarray
    pixels: int
    height_m: float
    pixel_roi: tuple
    source: str = "depth"
    confidence: float | None = None
    track_id: int | None = None

    def as_dict(self):
        return dict(
            low_m=self.low.tolist(),
            high_m=self.high.tolist(),
            dimensions_m=(self.high - self.low).tolist(),
            height_above_table_m=self.height_m,
            pixels=self.pixels,
            pixel_roi=[int(v) for v in self.pixel_roi],
            estimated=True,
            source=self.source,
            confidence=self.confidence,
            track_id=self.track_id,
        )


def paper_mask(xy, config):
    bed = corners(config, inset=False)
    if bed is None or config.get("task_mode") == "can_discard":
        low, high = (
            np.asarray(config["workspace_min_m"]),
            np.asarray(config["workspace_max_m"]),
        )
        return ((xy >= low[:2]) & (xy <= high[:2])).all(axis=-1)
    edge = np.roll(bed[:, :2], -1, axis=0) - bed[:, :2]
    delta = xy[..., None, :] - bed[:, :2]
    cross = edge[:, 0] * delta[..., 1] - edge[:, 1] * delta[..., 0]
    return (cross >= -1e-6).all(axis=-1) | (cross <= 1e-6).all(axis=-1)


def detect_boxes(
    depth,
    intrinsics,
    config,
    arm=None,
    q=None,
    min_height=0.018,
    min_pixels=100,
    stride=2,
    instance_mask=None,
    plane_bias_m=None,
):
    began = time.perf_counter()
    if not config.get("calibration_verified") or config.get("T_base_camera") is None:
        raise ValueError("Calibrate camera and table before object detection")
    depth = np.asarray(depth, float)
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]

    def val(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    camera = np.stack(
        ((u - val("ppx")) / val("fx") * z, (v - val("ppy")) / val("fy") * z, z), axis=-1
    )
    transform = np.asarray(config["T_base_camera"])
    points = camera @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(z) & (z > 0.1) & (z < 2)
    if instance_mask is None:
        valid &= paper_mask(points[..., :2], config)
    else:
        valid &= np.asarray(instance_mask, bool)[::stride, ::stride]
        # The operator's paper/playground defines the detection area too.
        valid &= paper_mask(points[..., :2], config)
        valid &= (
            (points[..., :2] >= np.array(config["workspace_min_m"][:2]))
            & (points[..., :2] <= np.array(config["workspace_max_m"][:2]))
        ).all(axis=-1)
    height = points[..., 2] - table_height_at(points[..., :2], config)
    # Estimate a segmentation-only plane bias from low tabletop observations.
    # This does not change the robot calibration or its table-clearance checks.
    bias = (
        float(np.percentile(height[valid], 15))
        if instance_mask is None and valid.any()
        else 0.0
    )
    bias = float(np.clip(bias, -0.015, 0.015))
    if plane_bias_m is not None:
        supplied = np.asarray(plane_bias_m)
        bias = (
            supplied[::stride, ::stride]
            if supplied.ndim == 2
            else float(np.clip(supplied, -0.04, 0.04))
        )
    raised = valid & (height - bias > min_height) & (height - bias < 0.40)
    if arm is not None and q is not None:
        links = arm.frames(q)[0]
        for index, (a, b) in enumerate(zip(links[:-1], links[1:])):
            delta = b - a
            fraction = np.clip(
                ((points - a) * delta).sum(-1) / max(delta @ delta, 1e-20), 0, 1
            )
            distance = np.linalg.norm(
                points - (a + fraction[..., None] * delta), axis=-1
            )
            radius = 0.055 if index < 2 else 0.032
            raised &= distance > radius
    mask = ndimage.binary_opening(raised, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    boxes = []
    for label in range(1, count + 1):
        region = (labels == label) & raised
        if region.sum() < min_pixels:
            continue
        cloud = points[region]
        low, high = np.percentile(cloud, [1, 99], axis=0)
        workspace_low = np.asarray(config["workspace_min_m"])
        workspace_high = np.asarray(config["workspace_max_m"])
        if np.any(high[:2] < workspace_low[:2]) or np.any(low[:2] > workspace_high[:2]):
            continue
        if min(high[:2] - low[:2]) < 0.008:
            continue
        # Keep the observed upper extent; the lower face sits on the fitted table.
        footprint = np.array(
            [[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]
        )
        local_bias = float(np.median(bias[region])) if np.ndim(bias) else bias
        low[2] = float(np.min(table_height_at(footprint, config) + local_bias))
        top = float(np.percentile((height - bias)[region], 99))
        rows, cols = np.nonzero(region)
        boxes.append(
            DetectedBox(
                low,
                high,
                int(region.sum() * stride * stride),
                top,
                (
                    int(cols.min() * stride),
                    int(rows.min() * stride),
                    int(cols.max() * stride),
                    int(rows.max() * stride),
                ),
            )
        )
    boxes.sort(key=lambda box: box.pixels, reverse=True)
    return boxes, dict(
        processing_ms=(time.perf_counter() - began) * 1000,
        plane_depth_bias_m=float(np.median(bias)),
        raised_pixels=int(raised.sum() * stride * stride),
        region="A4 tabletop" if config.get("test_bed") else "playground XY",
        robot_masked=arm is not None and q is not None,
    )


def draw_boxes(image, boxes, config, intrinsics):
    from PIL import ImageDraw

    from ..overlays import project
    from ..workspace_geometry import box_edges

    draw = ImageDraw.Draw(image)
    for index, box in enumerate(boxes, 1):
        bounds = dict(workspace_min_m=box.low, workspace_max_m=box.high)
        edges = box_edges(bounds) if box.source == "depth" else []
        for a, b in edges:
            p, q = (
                project(a, config["T_base_camera"], intrinsics),
                project(b, config["T_base_camera"], intrinsics),
            )
            if p is not None and q is not None:
                draw.line((p, q), fill="#ffba75", width=2)
        if box.source != "depth":
            draw.rectangle(box.pixel_roi, outline="#ffba75", width=2)
        dimensions = (box.high - box.low) * 100
        label = f"Box {index} {dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f} cm est."
        x, y = box.pixel_roi[:2]
        draw.text(
            (min(x, image.width - 235), max(45, y - 17)),
            label,
            fill="#ffcf9c",
            stroke_width=1,
            stroke_fill="#171719",
        )


def grouped_footprints(boxes, limit=2):
    rectangles = [np.r_[b.low[:2], b.high[:2]] for b in boxes]
    while len(rectangles) > limit:
        choices = []
        for i, a in enumerate(rectangles):
            for j in range(i + 1, len(rectangles)):
                b = rectangles[j]
                gap = np.maximum(np.maximum(a[:2] - b[2:], b[:2] - a[2:]), 0)
                choices.append((float(np.linalg.norm(gap)), i, j))
        _, i, j = min(choices)
        a, b = rectangles[i], rectangles[j]
        merged = np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])]
        rectangles = [r for k, r in enumerate(rectangles) if k not in (i, j)] + [merged]
    return sorted(rectangles, key=lambda r: tuple(r[:2]))
