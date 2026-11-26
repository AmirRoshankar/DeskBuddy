"""Known rectangular tabletop footprint, independent of robot motion."""

import numpy as np


def corners(config, inset=True):
    bed = config.get("test_bed")
    if bed is None:
        return None
    p = np.asarray(bed["corners_base_m"], float)
    if p.shape != (4, 3) or not np.isfinite(p).all():
        raise ValueError("Test bed needs four finite ordered XYZ corners")
    if config.get("assume_level_table"):
        width, length = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])
        x = p[1, :2] - p[0, :2]
        x = x / max(np.linalg.norm(x), 1e-12)
        y = np.array([-x[1], x[0]])
        if y @ (p[3, :2] - p[0, :2]) < 0:
            y = -y
        center = p[:, :2].mean(0)
        xy = np.array(
            [
                center - width * x / 2 - length * y / 2,
                center + width * x / 2 - length * y / 2,
                center + width * x / 2 + length * y / 2,
                center - width * x / 2 + length * y / 2,
            ]
        )
        p = np.c_[xy, np.full(4, config["table_z_m"])]
    a, b = p[1] - p[0], p[3] - p[0]
    width, length = np.linalg.norm(a), np.linalg.norm(b)
    if (
        min(width, length) < 0.05
        or abs(a @ b / (width * length)) > 0.05
        or np.linalg.norm(p[2] - (p[0] + a + b)) > 0.002
    ):
        raise ValueError("Test bed must be a rectangular planar footprint")
    margin = float(bed.get("inset_m", 0.02)) if inset else 0.0
    if not np.isfinite(margin) or margin < 0 or 2 * margin >= min(width, length):
        raise ValueError("Invalid test-bed inset")
    x, y = a / width, b / length
    return p + np.array(
        [
            margin * x + margin * y,
            -margin * x + margin * y,
            -margin * x - margin * y,
            margin * x - margin * y,
        ]
    )


def contains_xy(point, config):
    p = corners(config)
    if p is None:
        return True
    xy = np.asarray(point)[:2]
    edge = np.roll(p[:, :2], -1, axis=0) - p[:, :2]
    delta = xy - p[:, :2]
    cross = edge[:, 0] * delta[:, 1] - edge[:, 1] * delta[:, 0]
    return bool((cross >= -1e-9).all() or (cross <= 1e-9).all())


def table_height_at(xy, config):
    """Height of the fitted table plane at XY, including its calibration margin."""
    xy = np.asarray(xy, float)
    p = corners(config, inset=False)
    if p is None:
        return np.full(xy.shape[:-1], config["table_z_m"])
    normal = np.cross(p[1] - p[0], p[3] - p[0])
    if abs(normal[2]) < 0.1 * np.linalg.norm(normal):
        raise ValueError(
            "Fitted table plane is nearly vertical; check corner order and camera fit"
        )
    margin = config["test_bed"].get("clearance_margin_m", 0.003)
    return p[0, 2] - ((xy - p[0, :2]) @ normal[:2]) / normal[2] + margin


def table_clearance(points, config):
    points = np.asarray(points, float)
    margin = np.full(points.shape[:-1], config["link_clearance_m"])
    if config.get("grasp_contact_mode") and points.ndim == 2 and len(points) >= 2:
        margin[-2:] = 0.001
    return points[..., 2] - table_height_at(points[..., :2], config) - margin


def surface_point(u, v, intrinsics, config):
    p = corners(config, inset=False)
    if p is None:
        z = config["table_z_m"]
        p = np.array([[0, 0, z], [0.21, 0, z], [0.21, 0.297, z], [0, 0.297, z]])

    bed = config.get("test_bed") or {}
    if config.get("assume_level_table") and bed.get("corner_pixels") is not None:
        import cv2

        h = cv2.getPerspectiveTransform(
            np.float32(bed["corner_pixels"]), np.float32(p[:, :2])
        )
        xy = cv2.perspectiveTransform(np.array([[[u, v]]], np.float64), h)[0, 0]
        point = np.r_[xy, config["table_z_m"]]
        if config.get("restrict_to_test_bed", True) and not contains_xy(point, config):
            raise ValueError("Click inside the A4 selection grid")
        return point

    def value(key):
        return (
            intrinsics[key]
            if isinstance(intrinsics, dict)
            else getattr(intrinsics, key)
        )

    ray = np.array(
        [(u - value("ppx")) / value("fx"), (v - value("ppy")) / value("fy"), 1.0]
    )
    t = np.asarray(config["T_base_camera"])
    direction = t[:3, :3] @ ray
    normal = np.cross(p[1] - p[0], p[3] - p[0])
    denominator = normal @ direction
    if abs(denominator) < 1e-9:
        raise ValueError("View ray is parallel to the paper")
    distance = normal @ (p[0] - t[:3, 3]) / denominator
    if distance <= 0:
        raise ValueError("Paper is behind the camera")
    point = t[:3, 3] + distance * direction
    if config.get("restrict_to_test_bed", True) and not contains_xy(point, config):
        raise ValueError(
            "Click inside the A4 selection grid (2 cm inset from paper edges)"
        )
    return point
