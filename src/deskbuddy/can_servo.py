"""Camera-guided pre-grasp correction: measure jaw vs lid in the same images.

At the pregrasp pose (jaw a few centimetres above the lid) each calibrated
RGB view sees both the fingertips and the lid. Both features are located by
aligning their *predicted* projections to image edges, exactly like the lid
fit in trajectory.can_opening:

* lid shift   = where the lid really is vs. where the can estimate projects;
* finger shift = where the fingers really are vs. where encoder FK projects.

Converting each image shift to metres on its own horizontal plane and taking
the difference gives the jaw-to-lid error. A camera pose error moves both
features almost equally at this short distance, so it largely cancels; what
remains is the error the grasp actually needs fixed (can localization, tool
calibration, joint backlash). The grasp is then shifted by minus that error.

Safety gates: every view needs a sharp lid fit and finger fit, views must
agree, and the correction is capped. Otherwise nothing changes and the
measurement is only logged. No hardware access here; execution decides.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .test_bed import corners
from .trajectory.can_opening import fit_lid, lid_plane_homography
from .trajectory.rgb_can_perception import projector

MAX_CORRECTION_M = 0.015
MAX_VIEW_DISAGREEMENT_M = 0.005
MAX_EDGE_FIT_PX = 1.6
FINGER_BAND_M = 0.03
LOG_ROOT = Path("runs/servo-log")


def view_projectors(config, intrinsics, image_count):
    world = corners(config, inset=False)
    bed = config.get("test_bed")
    if world is None or bed is None:
        raise ValueError("Table calibration is required for camera guidance")
    views = [
        projector(config["T_base_camera"], intrinsics, world, bed["corner_pixels"])
    ]
    secondary = config.get("secondary_camera_calibration") or {}
    if (
        image_count > 1
        and secondary.get("T_base_camera")
        and secondary.get("intrinsics")
    ):
        views.append(
            projector(
                secondary["T_base_camera"],
                secondary["intrinsics"],
                world,
                secondary["secondary_pixels"],
            )
        )
    return views


def finger_points(arm, q, percent):
    """World vertices of the fixed and moving jaw within the lowest 3 cm."""
    from .self_collision import world_parts

    parts, _ = world_parts(arm, q, jaw_deg=percent)
    vertices = np.concatenate(
        [
            p["vertices"]
            for p in parts
            if p["name"] in ("gripper_link", "moving_jaw_so101_v1_link")
        ]
    )
    return vertices[vertices[:, 2] < vertices[:, 2].min() + FINGER_BAND_M]


def outline(pixels, samples=160):
    """Densely sampled convex outline of projected points."""
    hull = cv2.convexHull(np.float32(pixels)).reshape(-1, 2)
    closed = np.vstack([hull, hull[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(lengths)]
    along = np.linspace(0, distance[-1], samples, endpoint=False)
    return np.c_[
        np.interp(along, distance, closed[:, 0]),
        np.interp(along, distance, closed[:, 1]),
    ]


def plane_shift(project, xy, z, pixel, shift_px):
    """Image shift at ``pixel`` -> metres on the horizontal plane z near xy."""
    homography = lid_plane_homography(project, np.asarray(xy, float), z)
    a = cv2.perspectiveTransform(np.float32([[pixel]]), homography)[0, 0]
    b = cv2.perspectiveTransform(
        np.float32([[np.asarray(pixel) + shift_px]]), homography
    )[0, 0]
    return (b - a).astype(float)


def measure_view(image, project, arm, q, percent, can, debug=None):
    from .can_discard import jaw_center, lid_radius

    gray = cv2.cvtColor(np.asarray(image)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    base = np.asarray(can["center_m"], float)
    seam_z = base[2] + float(can["height_m"])
    ring = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    seam = np.c_[np.cos(ring), np.sin(ring)] * lid_radius(can) + base[:2]
    seam_px = project(np.c_[seam, np.full(len(seam), seam_z)])
    lid = fit_lid(gray, seam_px, search_px=30)

    fingers = finger_points(arm, q, percent)
    finger_px = outline(project(fingers))
    finger = fit_lid(gray, finger_px, search_px=25, scales=(1.0,))
    tip = jaw_center(arm, q, percent)
    tip_px = project(tip[None])[0]
    lid_px = lid["predicted_center_px"]
    lid_world = plane_shift(project, base[:2], seam_z, lid_px, lid["shift_px"])
    finger_world = plane_shift(project, tip[:2], tip[2], tip_px, finger["shift_px"])
    result = dict(
        lid_edge_fit_px=lid["chamfer_px"],
        finger_edge_fit_px=finger["chamfer_px"],
        lid_shift_px=list(map(float, lid["shift_px"])),
        finger_shift_px=list(map(float, finger["shift_px"])),
        lid_shift_m=lid_world.tolist(),
        finger_shift_m=finger_world.tolist(),
        # Jaw minus lid, relative to what the plan assumed.
        relative_error_m=(finger_world - lid_world).tolist(),
        usable=bool(
            lid["chamfer_px"] <= MAX_EDGE_FIT_PX
            and finger["chamfer_px"] <= MAX_EDGE_FIT_PX
        ),
    )
    if debug is not None:
        debug.append(
            dict(
                seam_px=seam_px,
                lid=lid,
                finger_px=finger_px,
                finger=finger,
                tip_px=tip_px,
            )
        )
    return result


def measure(images, intrinsics, config, arm, q, percent, can, debug=None):
    """Combine views into one XY correction to apply to the grasp target."""
    views = view_projectors(config, intrinsics, len(images))
    per_view = []
    for image, project in zip(images, views):
        try:
            per_view.append(measure_view(image, project, arm, q, percent, can, debug))
        except ValueError as exc:
            per_view.append(dict(usable=False, error=str(exc)))
    usable = [v for v in per_view if v["usable"]]
    report = dict(views=per_view, applied=False, unix_s=time.time())
    if not usable:
        report["reason"] = "No view had a sharp lid and finger fit"
        return report
    errors = np.array([v["relative_error_m"] for v in usable])
    if (
        len(errors) > 1
        and np.linalg.norm(errors[0] - errors[1]) > MAX_VIEW_DISAGREEMENT_M
    ):
        report["reason"] = (
            f"Views disagree by {np.linalg.norm(errors[0] - errors[1]) * 1000:.1f} mm"
        )
        return report
    error = errors.mean(0)
    report["relative_error_m"] = error.tolist()
    if np.linalg.norm(error) > MAX_CORRECTION_M:
        report["reason"] = (
            f"Measured {np.linalg.norm(error) * 1000:.0f} mm offset exceeds the "
            f"{MAX_CORRECTION_M * 1000:.0f} mm correction cap"
        )
        return report
    if len(usable) < 2:
        report["note"] = "single view; no cross-check"
    report.update(applied=True, correction_m=(-error).tolist())
    return report


def save_log(report, images, debug, folder=LOG_ROOT):
    """Images + measurements for later learning of a correction model."""
    folder = Path(folder) / time.strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    for index, (image, item) in enumerate(zip(images, debug)):
        out = np.asarray(image)[..., :3].copy()
        for points, color in (
            (item["seam_px"], (90, 120, 255)),
            (
                (item["seam_px"] - item["lid"]["predicted_center_px"])
                * item["lid"]["scale"]
                + item["lid"]["predicted_center_px"]
                + item["lid"]["shift_px"],
                (240, 200, 90),
            ),
            (item["finger_px"], (255, 90, 90)),
            (item["finger_px"] + item["finger"]["shift_px"], (90, 255, 120)),
        ):
            cv2.polylines(out, [np.int32(points)], True, color, 1)
        cv2.imwrite(
            str(folder / f"view-{index}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        )
    serializable = json.loads(
        json.dumps(report, default=lambda v: np.asarray(v).tolist())
    )
    (folder / "measurement.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    return folder


def check_live(config_path="configs/local.json", live="runs/live"):
    """Offline check on the viewer's latest raw frames and encoder pose."""
    from PIL import Image

    from .can_discard import detect_can, opening_for_width
    from .config import load_config
    from .kinematics import Arm
    from .trajectory.perception import DetectedBox

    config = load_config(config_path)
    state = json.loads((Path(live) / "state.json").read_text(encoding="utf-8"))
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    q = np.asarray(state["motor_state"]["q_model_rad"])
    percent = float(state["motor_state"].get("gripper_percent", 0.0))
    boxes = [
        DetectedBox(
            np.array(b["low_m"]),
            np.array(b["high_m"]),
            b.get("pixels", 0),
            b.get("height_above_table_m", 0),
            tuple(b.get("pixel_roi", [0, 0, 0, 0])),
            confidence=b.get("confidence"),
        )
        for b in state.get("obstacle_boxes", [])
    ]
    can = detect_can(boxes, config)
    images = []
    for name in state.get("pickup_rgb_paths") or []:
        path = Path(name)
        if not path.is_file():
            path = Path(live) / name.replace("\\", "/").rsplit("/", 1)[-1]
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
    debug = []
    report = measure(images, state["intrinsics"], config, arm, q, percent, can, debug)
    report["gripper_percent"] = percent
    report["opening_for_8mm"] = opening_for_width(arm, 0.008)
    folder = save_log(report, images, debug)
    return report, folder


def main():
    report, folder = check_live()
    print(json.dumps(report, indent=2, default=lambda v: np.asarray(v).tolist()))
    print(f"Annotated views: {folder}")


if __name__ == "__main__":
    main()
