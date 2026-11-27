"""Webcam-only fingertip/target image error at a known hover height."""

import cv2
import numpy as np

from .can_servo import MAX_EDGE_FIT_PX, finger_points, outline
from .test_bed import corners
from .trajectory.can_opening import detect_webcam_opening, fit_lid
from .trajectory.rgb_can_perception import projector


def measurement(
    image, config, arm, q, percent, can, desired_center, hole=False, tracked_tip=None
):
    from .can_discard import jaw_center, lid_radius

    calibration = config.get("secondary_camera_calibration") or {}
    world = corners(config, inset=False)
    project = projector(
        calibration["T_base_camera"],
        calibration["intrinsics"],
        world,
        calibration["secondary_pixels"],
    )
    gray = cv2.cvtColor(np.asarray(image)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if tracked_tip is None:
        finger = fit_lid(
            gray,
            outline(project(finger_points(arm, q, percent))),
            search_px=20,
            scales=(1.0,),
        )
        if finger["chamfer_px"] > MAX_EDGE_FIT_PX:
            return None
    else:
        finger = dict(chamfer_px=0.0, shift_px=[0.0, 0.0])
    center = jaw_center(arm, q, percent)
    desired = np.asarray(desired_center, float).copy()
    desired[2] = center[
        2
    ]  # lateral alignment only, with model-based height compensation
    if hole:
        detected = detect_webcam_opening(image, config, can)
        target = np.asarray(detected["center_m"], float)
        # Preserve the planned jaw-center offset relative to the hole; orange
        # paint is not assumed to identify the fingertip/contact location.
        nominal = np.asarray(
            (config.get("can_request") or {})
            .get("model_proposal", {})
            .get("hole_center_m", target),
            float,
        )
        desired[:2] += target[:2] - nominal[:2]
        target_shift = np.zeros(2)
    else:
        base = np.asarray(can["center_m"], float)
        angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        ring = np.c_[np.cos(angle), np.sin(angle)] * lid_radius(can) + base[:2]
        seam = project(np.c_[ring, np.full(len(ring), base[2] + can["height_m"])])
        lid = fit_lid(gray, seam, search_px=20)
        if lid["chamfer_px"] > MAX_EDGE_FIT_PX:
            return None
        target_shift = np.asarray(lid["shift_px"], float)
    if tracked_tip is not None:
        from .can_discard import fingertips

        fixed = fingertips(arm, q, percent)[0]
        desired += fixed - center
        observed = np.asarray(tracked_tip, float)
    else:
        observed = project(center[None])[0] + finger["shift_px"]
    target_pixel = project(desired[None])[0] + target_shift
    epsilon = 0.001
    jacobian = np.column_stack(
        [
            (
                project((center + np.eye(3)[j] * epsilon)[None])[0]
                - project(center[None])[0]
            )
            / epsilon
            for j in (0, 1)
        ]
    )
    return dict(
        error_px=(observed - target_pixel).tolist(),
        jacobian_px_m=jacobian.tolist(),
        finger_fit_px=float(finger["chamfer_px"]),
        estimated_height_m=float(center[2]),
    )


def read_frame(read_live, newer_than):
    """Read only the webcam snapshot; a missing/stale RealSense does not veto it."""
    import time
    from pathlib import Path

    from PIL import Image

    report = read_live() or {}
    stamp = report.get("pickup_rgb_unix_s", 0)
    paths = report.get("pickup_rgb_paths") or []
    if stamp <= newer_than or time.time() - stamp > 2.5 or len(paths) < 2:
        return None
    path = Path(paths[1])
    if abs(path.stat().st_mtime - stamp) > 0.5:
        return None
    with Image.open(path) as image:
        return stamp, np.asarray(image.convert("RGB")), report.get("tip_tracking")
