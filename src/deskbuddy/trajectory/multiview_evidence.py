"""Cross-view instance confirmation; depth stays in the calibrated RGB-D base frame."""

import itertools

import numpy as np

from ..multiview import project_secondary


def confirm_box(box, instances, calibration, robot_mask=None):
    import cv2

    if calibration.get("T_base_camera") and calibration.get("intrinsics"):
        pixels = [
            project_secondary(p, calibration, allow_planar=True)
            for p in itertools.product(*zip(box.low, box.high))
        ]
        pixels = [p for p in pixels if p is not None]
        if len(pixels) < 4:
            return False, "outside webcam"
        width, height = calibration["image_size"]
        silhouette = np.zeros((height, width), np.uint8)
        hull = cv2.convexHull(np.int32(pixels))
        cv2.fillConvexPoly(silhouette, hull, 1)
        silhouette = cv2.dilate(silhouette, np.ones((11, 11), np.uint8)) > 0
        if (
            robot_mask is not None
            and (silhouette & robot_mask).sum() / max(silhouette.sum(), 1) > 0.4
        ):
            return False, "occluded by arm"
        for instance in instances:
            mask = instance["mask"]
            overlap = (silhouette & mask).sum() / max(
                min(silhouette.sum(), mask.sum()), 1
            )
            if overlap > 0.3:
                return True, "two-view 3D silhouette"
        return False, "not confirmed in webcam"
    # A planar homography is used only for the object's visible base-contact edge.
    # Do not map its raised center to the table or invent a second depth reading.
    inverse = np.linalg.inv(calibration["H_secondary_paper"])
    corners = np.asarray(calibration["corners_base_m"])
    xaxis, yaxis = (corners[1] - corners[0]) / 0.210, (corners[3] - corners[0]) / 0.297
    for instance in instances:
        v, u = np.nonzero(instance["mask"])
        if len(u) < 10:
            continue
        bottom = v >= np.percentile(v, 95)
        xy = cv2.perspectiveTransform(
            np.c_[u[bottom], v[bottom]].astype(np.float32)[None], inverse
        )[0]
        points = corners[0] + xy[:, :1] * xaxis + xy[:, 1:] * yaxis
        near = (
            (points[:, :2] >= box.low[:2] - 0.02)
            & (points[:, :2] <= box.high[:2] + 0.02)
        ).all(axis=1)
        if near.mean() > 0.3:
            return True, "two-view base-contact support (planar)"
    return False, "base contact not confirmed in webcam"


def secondary_instances(model, rgb, device, calibration=None):

    import torch

    result = model.predict(
        np.ascontiguousarray(rgb[..., ::-1]),
        device=device,
        imgsz=640,
        conf=0.28,
        retina_masks=True,
        quantize=16 if torch.cuda.is_available() and device != "cpu" else 32,
        verbose=False,
        max_det=25,
        agnostic_nms=True,
    )[0]
    instances = instances_from_result(result)
    if calibration is None:
        return instances
    return filter_contact_instances(instances, calibration)


def instances_from_result(result):
    masks = result.masks.data.cpu().numpy() if result.masks is not None else []
    return [
        dict(
            mask=mask > 0.5,
            roi=[int(v) for v in detection.xyxy[0].cpu().numpy()],
            confidence=float(detection.conf.item()),
        )
        for detection, mask in zip(result.boxes, masks)
        if 0.001 < (mask > 0.5).mean() < 0.45
    ]


def filter_contact_instances(instances, calibration):
    import cv2

    inverse = np.linalg.inv(calibration["H_secondary_paper"])
    accepted = []
    for item in instances:
        v, u = np.nonzero(item["mask"])
        bottom = v >= np.percentile(v, 95)
        xy = cv2.perspectiveTransform(
            np.c_[u[bottom], v[bottom]].astype(np.float32)[None], inverse
        )[0]
        inside = ((xy >= [-0.015, -0.015]) & (xy <= [0.225, 0.312])).all(axis=1)
        if inside.mean() > 0.25:
            accepted.append(item)
    return accepted


def raised_cloud(depth, intrinsics, config, robot_mask=None, plane_bias=None, stride=2):
    """Measured foreground only: same table, paper, robot and XY bounds in both views."""
    from ..test_bed import table_height_at
    from .perception import paper_mask

    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = np.asarray(depth)[::stride, ::stride]

    def val(k):
        return intrinsics[k] if isinstance(intrinsics, dict) else getattr(intrinsics, k)

    camera = np.stack(
        ((u - val("ppx")) * z / val("fx"), (v - val("ppy")) * z / val("fy"), z), -1
    )
    t = np.asarray(config["T_base_camera"])
    world = camera @ t[:3, :3].T + t[:3, 3]
    height = world[..., 2] - table_height_at(world[..., :2], config)
    bias = 0 if plane_bias is None else np.asarray(plane_bias)
    if np.ndim(bias):
        bias = bias[::stride, ::stride]
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    valid = np.isfinite(z) & (z > 0.1) & (z < 2)
    valid &= paper_mask(world[..., :2], config)
    valid &= ((world[..., :2] >= lo[:2]) & (world[..., :2] <= hi[:2])).all(-1)
    valid &= (height - bias > 0.018) & (height - bias < 0.4)
    if robot_mask is not None:
        valid &= ~robot_mask[::stride, ::stride]
    return world[valid], np.c_[u[valid], v[valid]]


def project_cloud(points, calibration):
    """Vectorized projection; estimated poses remain explicitly unverified."""
    t = np.linalg.inv(np.asarray(calibration["T_base_camera"]))
    camera = points @ t[:3, :3].T + t[:3, 3]
    k = calibration["intrinsics"]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.c_[
            camera[:, 0] / camera[:, 2] * k["fx"] + k["ppx"],
            camera[:, 1] / camera[:, 2] * k["fy"] + k["ppy"],
        ]
    valid = np.isfinite(uv).all(1) & (camera[:, 2] > 0.03)
    return uv, valid


def filter_secondary(instances, calibration, config, cloud=None):
    """Discard sheet-sized masks; show only raised world-supported RGB fragments.

    The webcam has no depth. Its raw YOLO rectangles are never 3D measurements.
    """
    import cv2

    from .perception import paper_mask

    corners = np.float32(calibration["secondary_pixels"])
    width, height = calibration["image_size"]
    paper = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(paper, np.int32(corners), 1)
    # Workspace on the table; raised support below handles elevated silhouettes.
    plane = np.asarray(calibration["corners_base_m"])
    h = np.asarray(calibration["H_secondary_paper"])
    v, u = np.mgrid[0:height, 0:width]
    xy = cv2.perspectiveTransform(
        np.c_[u.ravel(), v.ravel()].astype(np.float32)[None], np.linalg.inv(h)
    )[0]
    world = (
        plane[0]
        + xy[:, :1] * (plane[1] - plane[0]) / 0.210
        + xy[:, 1:] * (plane[3] - plane[0]) / 0.297
    )
    lo, hi = np.array(config["workspace_min_m"]), np.array(config["workspace_max_m"])
    roi = (
        paper_mask(world[:, :2], config)
        & ((world[:, :2] >= lo[:2]) & (world[:, :2] <= hi[:2])).all(1)
    ).reshape(height, width)
    support = None
    projected = None
    if (
        cloud is not None
        and calibration.get("T_base_camera")
        and calibration.get("intrinsics")
    ):
        uv, valid = project_cloud(cloud, calibration)
        inside = (
            valid
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        support = np.zeros((height, width), np.uint8)
        rounded = np.rint(uv[inside]).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        support[rounded[:, 1], rounded[:, 0]] = 1
        support = cv2.dilate(support, np.ones((19, 19), np.uint8)) > 0
        projected = (uv, valid)
    accepted = []
    for item in instances:
        mask = item["mask"].copy()
        # Paper prediction follows the full known quadrilateral, not an object.
        intersection = (mask & (paper > 0)).sum()
        iou = intersection / max((mask | (paper > 0)).sum(), 1)
        if iou > 0.6:
            continue
        if support is not None:
            mask &= support
        else:
            mask &= roi
        if mask.sum() < 25:
            continue
        v, u = np.nonzero(mask)
        accepted.append(
            dict(
                item,
                mask=mask,
                roi=[int(u.min()), int(v.min()), int(u.max()), int(v.max())],
            )
        )
    return accepted, projected


def fuse_fragments(boxes, instances, calibration, cloud, primary_pixels, projected):
    """One world box per associated object; grow only through measured nearby points.

    Both views must support a seed. A common webcam mask can merge nearby partial
    primary boxes, but never joins distant objects just because a mask spans them.
    """
    import copy

    import cv2
    from scipy.spatial import cKDTree

    if projected is None:
        return boxes
    uv, valid = projected
    width, height = calibration["image_size"]
    inside = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    indices = np.flatnonzero(inside)
    pixels = np.floor(uv[inside]).astype(int)
    consumed = set()
    output = []
    for instance in instances:
        mask = (
            cv2.dilate(instance["mask"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        )
        supported = indices[mask[pixels[:, 1], pixels[:, 0]]]
        if len(supported) < 15:
            continue
        groups = []
        for i, box in enumerate(boxes):
            if i in consumed:
                continue
            near = (
                (cloud[supported] >= box.low - 0.015)
                & (cloud[supported] <= box.high + 0.015)
            ).all(1)
            if near.sum() >= 8:
                groups.append(i)
        while groups:
            group = [groups.pop(0)]
            changed = True
            while changed:
                changed = False
                for i in groups[:]:
                    if any(
                        np.linalg.norm(
                            np.maximum(
                                np.maximum(
                                    boxes[i].low - boxes[j].high,
                                    boxes[j].low - boxes[i].high,
                                ),
                                0,
                            )
                        )
                        < 0.025
                        for j in group
                    ):
                        group.append(i)
                        groups.remove(i)
                        changed = True
            lo = np.min([boxes[i].low for i in group], axis=0)
            hi = np.max([boxes[i].high for i in group], axis=0)
            near = (
                (cloud[supported] >= lo - 0.035) & (cloud[supported] <= hi + 0.035)
            ).all(1)
            measured = supported[near]
            if len(measured) < 15:
                continue
            # Bound expansion to connected surface neighborhoods, not every point
            # behind a broad projection in the opposite view.
            seed = (
                (cloud[measured] >= lo - 0.01) & (cloud[measured] <= hi + 0.01)
            ).all(1)
            tree = cKDTree(cloud[measured])
            active = set(np.flatnonzero(seed).tolist())
            front = list(active)
            for _ in range(8):
                if not front:
                    break
                neighbors = set(
                    j
                    for row in tree.query_ball_point(cloud[measured][front], 0.012)
                    for j in row
                )
                front = list(neighbors - active)
                active.update(front)
            if not active:
                continue
            measured = measured[list(active)]
            result = copy.deepcopy(boxes[group[0]])
            bounds = np.percentile(cloud[measured], [1, 99], axis=0)
            result.low = np.minimum(lo, bounds[0])
            result.high = np.maximum(hi, bounds[1])
            roi = primary_pixels[measured]
            result.pixel_roi = (
                min(int(roi[:, 0].min()), *(boxes[i].pixel_roi[0] for i in group)),
                min(int(roi[:, 1].min()), *(boxes[i].pixel_roi[1] for i in group)),
                max(int(roi[:, 0].max()), *(boxes[i].pixel_roi[2] for i in group)),
                max(int(roi[:, 1].max()), *(boxes[i].pixel_roi[3] for i in group)),
            )
            result.source = "Two-view RGB + measured RealSense foreground"
            result.height_m = float(result.high[2] - result.low[2])
            result.pixels = len(measured) * 4
            output.append(result)
            consumed.update(group)
    return output
