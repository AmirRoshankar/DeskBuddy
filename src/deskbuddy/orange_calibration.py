"""Orange calibration marker detection; no motor or camera access."""

import cv2
import numpy as np


def detect_orange_marker(rgb, previous=None, initial_hint=None, search_radius=None):
    """Return a stable blob centroid, or None when colour evidence is ambiguous."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB camera image")
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([3, 120, 75]), np.array([25, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if area < 30 or area > image.shape[0] * image.shape[1] * 0.05:
            continue
        if width < 4 or height < 4 or area / (width * height) < 0.3:
            continue
        candidates.append(
            dict(
                pixel=centroids[i].tolist(),
                area=int(area),
                roi=[int(x), int(y), int(x + width), int(y + height)],
            )
        )
    if previous is not None or initial_hint is not None:
        hints = np.asarray(
            [previous] if previous is not None else initial_hint, float
        ).reshape(-1, 2)
        if len(hints) == 0:
            return None
        radius = (
            60.0
            if previous is not None
            else max(60.0, 0.18 * np.hypot(*image.shape[:2]))
        )
        if search_radius is not None:
            radius = float(search_radius)
        ranked = sorted(
            (float(np.linalg.norm(hints - np.asarray(c["pixel"]), axis=1).min()), i)
            for i, c in enumerate(candidates)
        )
        if not ranked or ranked[0][0] > radius:
            return None
        chosen = dict(candidates[ranked[0][1]])
        chosen["selection"] = (
            "nearest tracked tip"
            if previous is not None
            else "nearest sheet-center column"
        )
        return chosen
    if len(candidates) != 1:
        return None
    return candidates[0]


def sheet_center_hints(config, intrinsics):
    """Project a height column over the paper middle, independent of depth."""
    from .trajectory.rgb_can_perception import projector

    bed = config.get("test_bed") or {}
    second = config.get("secondary_camera_calibration") or {}
    corners = bed.get("corners_base_m")
    if corners is None:
        return [None, None]
    corners = np.asarray(corners, float)
    middle = corners.mean(axis=0)
    column = np.tile(middle, (16, 1))
    column[:, 2] += np.linspace(0.025, 0.30, 16)
    hints = []
    for pose, k, pixels in (
        (config.get("T_base_camera"), intrinsics, bed.get("corner_pixels")),
        (
            second.get("T_base_camera"),
            second.get("intrinsics"),
            second.get("secondary_pixels"),
        ),
    ):
        if pose is None or k is None or pixels is None:
            hints.append(None)
            continue
        projected = projector(pose, k, corners, pixels)(column)
        hints.append(projected[np.isfinite(projected).all(axis=1)].tolist())
    return hints


def triangulate_marker(first_pixel, second_pixel, first_camera, second_camera):
    """Triangulate two matched marker centers; reject weak/inconsistent geometry.

    Camera dictionaries contain calibrated intrinsics and T_base_camera. This
    measures the marker center, not the jaw contact point.
    """
    projections = []
    origins = []
    rays = []
    for pixel, camera in ((first_pixel, first_camera), (second_pixel, second_camera)):
        k = camera["intrinsics"]
        matrix = np.array([[k["fx"], 0, k["ppx"]], [0, k["fy"], k["ppy"]], [0, 0, 1.0]])
        transform = np.asarray(camera["T_base_camera"], float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Camera transform is invalid")
        base_to_camera = np.linalg.inv(transform)
        projections.append(matrix @ base_to_camera[:3])
        origins.append(transform[:3, 3])
        ray = transform[:3, :3] @ np.linalg.solve(matrix, [*pixel, 1])
        rays.append(ray / np.linalg.norm(ray))
    angle = np.arccos(np.clip(abs(np.dot(*rays)), 0, 1))
    if angle < np.deg2rad(5):
        raise ValueError("Camera rays too parallel for reliable marker depth")
    homogeneous = cv2.triangulatePoints(
        projections[0],
        projections[1],
        np.float64(first_pixel).reshape(2, 1),
        np.float64(second_pixel).reshape(2, 1),
    ).ravel()
    if abs(homogeneous[3]) < 1e-9:
        raise ValueError("Marker triangulation is degenerate")
    point = homogeneous[:3] / homogeneous[3]
    errors = []
    for pixel, camera, projection in zip(
        (first_pixel, second_pixel), (first_camera, second_camera), projections
    ):
        t = np.asarray(camera["T_base_camera"])
        if ((point - t[:3, 3]) @ t[:3, :3])[2] <= 0:
            raise ValueError("Marker lies behind a camera")
        projected = projection @ np.r_[point, 1.0]
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - pixel)))
    if max(errors) > 3:
        raise ValueError("Camera marker detections disagree; check calibration")
    return dict(
        point_m=point.tolist(),
        ray_angle_deg=float(np.rad2deg(angle)),
        reprojection_error_px=errors,
    )


def fit_arm_marker(arm, joint_samples, marker_points, heldout=3):
    """Fit bounded joint-zero corrections and a rigid marker position.

    Points must be independently triangulated in the calibrated robot-base
    frame. The marker is not assumed to be the gripper contact point. The last
    observations are held out and never enter the optimization.
    """
    from scipy.optimize import least_squares

    joints = np.asarray(joint_samples, float)
    observed = np.asarray(marker_points, float)
    if joints.ndim != 2 or joints.shape[1] != 5 or observed.shape != (len(joints), 3):
        raise ValueError("Expected paired five-joint poses and 3D marker observations")
    if len(joints) < 12 or heldout < 3 or heldout >= len(joints) - 8:
        raise ValueError("Need at least 12 poses including three independent checks")
    if not np.isfinite(joints).all() or not np.isfinite(observed).all():
        raise ValueError("Calibration observations must be finite")
    train = len(joints) - heldout
    if (
        np.linalg.svd(observed[:train] - observed[:train].mean(0), compute_uv=False)[-1]
        < 0.015
    ):
        raise ValueError("Marker poses need more 3D variation")

    def predict(parameters, samples):
        offsets = np.r_[parameters[:4], 0.0]
        local = parameters[4:]
        result = []
        for q in samples:
            _, transform = arm.frames(q + offsets)
            result.append(transform[:3, 3] + transform[:3, :3] @ local)
        return np.asarray(result)

    def residual(parameters):
        data = (predict(parameters, joints[:train]) - observed[:train]).ravel() / 0.003
        # Soft priors reduce overfitting; wrist-roll zero is fixed to remove an
        # unobservable rotation/marker-offset gauge.
        return np.r_[data, parameters[:4] / np.deg2rad(5), parameters[4:] / 0.08]

    bounds = (
        np.r_[np.full(4, -np.deg2rad(12)), np.full(3, -0.12)],
        np.r_[np.full(4, np.deg2rad(12)), np.full(3, 0.12)],
    )
    fit = least_squares(
        residual, np.zeros(7), bounds=bounds, loss="soft_l1", max_nfev=250
    )
    train_error = np.linalg.norm(
        predict(fit.x, joints[:train]) - observed[:train], axis=1
    )
    held_error = np.linalg.norm(
        predict(fit.x, joints[train:]) - observed[train:], axis=1
    )
    singular = np.linalg.svd(fit.jac[: train * 3], compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return dict(
        joint_offset_correction_deg=np.rad2deg(np.r_[fit.x[:4], 0.0]).tolist(),
        marker_offset_m=fit.x[4:].tolist(),
        train_error_mm=(train_error * 1000).tolist(),
        heldout_error_mm=(held_error * 1000).tolist(),
        condition=condition,
        passed=bool(fit.success and held_error.max() <= 0.008 and condition < 1000),
        note="Marker is not the jaw tip; independent camera/base accuracy is still required.",
    )


def track_orange_marker(previous_rgb, rgb, pixel):
    """Track the same feature with forward/backward optical-flow validation."""
    previous = cv2.cvtColor(
        np.asarray(previous_rgb).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    current = cv2.cvtColor(np.asarray(rgb).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if previous.shape != current.shape:
        return None
    start = np.float32(pixel).reshape(1, 1, 2)
    options = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, start, None, **options)
    if forward is None or not bool(ok[0, 0]):
        return None
    backward, ok, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, **options
    )
    if backward is None or not bool(ok[0, 0]) or np.linalg.norm(backward - start) > 1.5:
        return None
    return detect_orange_marker(rgb, previous=forward.ravel(), search_radius=20)
