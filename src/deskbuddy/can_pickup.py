"""RGB evidence that a can followed the jaw; no hardware API or depth input.

This is an appearance/motion check, not a force sensor. Ambiguous observations
return None so a possibly held can is never opened merely to try again.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .trajectory.rgb_can_perception import cylinder, projector


def match_motion(image, template, original, expected):
    """Track textured can-body pixels near the original/held hypotheses."""
    original, expected = np.asarray(original), np.asarray(expected)
    travel = np.linalg.norm(expected - original)
    if travel < 6:
        return None
    margin = max(template.shape[:2]) * 0.8
    low = np.maximum(np.minimum(original, expected) - margin, 0).astype(int)
    high = np.minimum(
        np.maximum(original, expected) + margin, [image.shape[1], image.shape[0]]
    ).astype(int)
    crop = image[low[1] : high[1], low[0] : high[0]]
    best = None
    for scale in (0.85, 1.0, 1.15):
        resized = cv2.resize(template, None, fx=scale, fy=scale)
        h, w = resized.shape[:2]
        if h >= crop.shape[0] or w >= crop.shape[1] or min(h, w) < 8:
            continue
        result = cv2.matchTemplate(crop, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, where = cv2.minMaxLoc(result)
        center = low + where + np.array([w, h]) / 2
        if best is None or score > best[0]:
            best = score, center
    if best is None or best[0] < 0.75:
        return None
    stationary = np.linalg.norm(best[1] - original)
    followed = np.linalg.norm(best[1] - expected)
    if followed < 0.45 * travel and stationary > 0.65 * travel:
        return True
    if stationary < max(3, 0.18 * travel) and followed > 0.7 * travel:
        return False
    return None


def lid_ring(can, samples=120):
    """Seam circle of the can lid in world coordinates."""
    from .can_discard import lid_radius

    center = np.asarray(can["center_m"], float)
    angle = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return np.c_[
        center[0] + lid_radius(can) * np.cos(angle),
        center[1] + lid_radius(can) * np.sin(angle),
        np.full(samples, center[2] + float(can["height_m"])),
    ]


def lid_vote(image, on_table, lifted, search=4, sharp=1.6, margin=0.8):
    """Is the lid edge where a held can would be, or still on the table?

    Texture-free: the silver seam ellipse is a strong edge in both cameras.
    Each hypothesis gets the best truncated median edge distance within a
    few pixels (calibration slack); the median tolerates the jaw covering
    part of the seam. Returns True (followed), False (stayed) or None.
    """
    on_table, lifted = np.asarray(on_table), np.asarray(lifted)
    if np.linalg.norm(on_table.mean(0) - lifted.mean(0)) < 6:
        return None
    gray = cv2.cvtColor(np.asarray(image)[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = gray.shape

    def cost(points):
        best = None
        for dx in range(-search, search + 1):
            for dy in range(-search, search + 1):
                p = np.int32(np.round(points + [dx, dy]))
                inside = (
                    (p[:, 0] >= 0)
                    & (p[:, 0] < width)
                    & (p[:, 1] >= 0)
                    & (p[:, 1] < height)
                )
                if inside.mean() < 0.6:
                    continue
                value = float(
                    np.median(np.minimum(distance[p[inside, 1], p[inside, 0]], 6))
                )
                best = value if best is None else min(best, value)
        return 6.0 if best is None else best

    table, held = cost(on_table), cost(lifted)
    if held <= sharp and table >= held + margin:
        return True
    if table <= sharp and held >= table + margin:
        return False
    return None


def read_images(read_live, newer_than=0):
    report = read_live()
    stamp = report.get("pickup_rgb_unix_s", 0)
    if stamp <= newer_than or time.time() - stamp > 2.5 or report.get("camera_stale"):
        return None
    paths = report.get("pickup_rgb_paths") or []
    if not paths:
        return None
    images = []
    for name in paths:
        path = Path(name)
        if abs(path.stat().st_mtime - stamp) > 0.5:
            return None
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")))
    return stamp, images, report


class PickupVerifier:
    def __init__(self, config, can, read_live, stopped):
        from .test_bed import corners

        if read_live is None:
            raise RuntimeError("Fresh RGB pickup verification is required")
        self.read_live, self.stopped = read_live, stopped
        self.can = can
        self.views = []
        frames = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not stopped():
            try:
                frames = read_images(read_live)
            except (OSError, ValueError):
                pass
            if frames is not None:
                break
            time.sleep(0.05)
        if stopped():
            raise RuntimeError("Can sequence stopped before verification capture")
        if frames is None:
            raise RuntimeError("Wait for fresh RGB pickup images before grasping")
        self.stamp, images, report = frames
        world = corners(config, inset=False)
        specs = [
            (
                config["T_base_camera"],
                report["intrinsics"],
                config["test_bed"]["corner_pixels"],
            )
        ]
        secondary = config.get("secondary_camera_calibration")
        if secondary and len(images) > 1:
            specs.append(
                (
                    secondary["T_base_camera"],
                    secondary["intrinsics"],
                    secondary["secondary_pixels"],
                )
            )
        points = cylinder(np.asarray(can["center_m"][:2]), can["center_m"][2])
        self.projects = []
        for index, (image, (transform, intrinsics, pixels)) in enumerate(
            zip(images, specs)
        ):
            project = projector(transform, intrinsics, world, pixels)
            self.projects.append((index, project))
            projected = project(points)
            low, high = projected.min(0), projected.max(0)
            # Inner body patch avoids the jaw, bright rim and background edges.
            a = np.maximum(low + (high - low) * [0.2, 0.3], 0).astype(int)
            b = np.minimum(
                low + (high - low) * [0.8, 0.85], [image.shape[1], image.shape[0]]
            ).astype(int)
            patch = image[a[1] : b[1], a[0] : b[0]].copy()
            if (
                min(patch.shape[:2]) < 12
                or patch.reshape(-1, 3).std(axis=0).mean() < 10
            ):
                continue
            center = (a + b) / 2
            self.views.append(
                (index, project, patch, center, (center - low) / (high - low))
            )
        if not self.views and not self.projects:
            raise RuntimeError("No calibrated view for the pickup check")

    def check(self, arm, grasp_pose, lifted_pose, closed, timeout=4.5):
        from .can_discard import jaw_center

        def rotation(q):
            return arm.frames(q)[1][:3, :3] @ arm.chain[-1][1][:3, :3].T

        def carried(points):
            relative = (points - jaw_center(arm, grasp_pose, closed)) @ rotation(
                grasp_pose
            )
            return relative @ rotation(lifted_pose).T + jaw_center(
                arm, lifted_pose, closed
            )

        points = cylinder(np.asarray(self.can["center_m"][:2]), self.can["center_m"][2])
        held = carried(points)
        seam = lid_ring(self.can)
        lid_hypotheses = [
            (index, project(seam), project(carried(seam)))
            for index, project in self.projects
        ]
        expectations = []
        for index, project, patch, original, fraction in self.views:
            projected = project(held)
            low, high = projected.min(0), projected.max(0)
            expectations.append((index, patch, original, low + (high - low) * fraction))
        after = time.time()
        previous = None
        confirmations = 0
        stamp = self.stamp
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.stopped():
                raise RuntimeError(
                    "Can sequence stopped during pickup check; hold retained"
                )
            try:
                frames = read_images(self.read_live, max(stamp, after))
            except (OSError, ValueError):
                frames = None
            if frames is None:
                time.sleep(0.05)
                continue
            stamp, images, _ = frames
            votes = [
                match_motion(images[index], patch, original, expected)
                for index, patch, original, expected in expectations
                if index < len(images)
            ] + [
                lid_vote(images[index], table, lifted)
                for index, table, lifted in lid_hypotheses
                if index < len(images)
            ]
            known = [vote for vote in votes if vote is not None]
            decision = known[0] if known and all(v == known[0] for v in known) else None
            confirmations = (
                confirmations + 1
                if decision is not None and decision == previous
                else int(decision is not None)
            )
            previous = decision
            if confirmations >= 2:
                return decision
        return None
