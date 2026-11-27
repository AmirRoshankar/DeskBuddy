"""RGB instance detection; depth estimates geometry only inside YOLO masks.

Class names are deliberately not used as collision semantics: a plain carton
can be classified as a remote. Missing depth retains a recent matched estimate.
This is an experimental preview map, not full-arm collision certification.
"""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .object_tracker import ObjectTracker
from .paper_surface import paper_surface
from .perception import detect_boxes
from .scene_stability import StableBoxes, image_changed, image_signature


def mask_grasp_pixel(mask):
    from scipy.ndimage import distance_transform_edt

    distances = distance_transform_edt(mask > 0.5)
    y, x = np.unravel_index(np.argmax(distances), distances.shape)
    return [int(x), int(y)]


def key_overlap(a, b):
    a, b = np.array(a), np.array(b)
    intersection = np.maximum(
        np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0
    ).prod()
    union = (
        np.maximum(a[2:] - a[:2], 0).prod()
        + np.maximum(b[2:] - b[:2], 0).prod()
        - intersection
    )
    return intersection / max(union, 1)


def model_is_real(model):
    return type(model).__module__.startswith("ultralytics.")


def roi_contains_key(roi, key):
    a, b = np.array(roi), np.array(key)
    area_a = np.maximum(a[2:] - a[:2], 0).prod()
    area_b = np.maximum(b[2:] - b[:2], 0).prod()
    intersection = np.maximum(
        np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0
    ).prod()
    return area_a < area_b * 2.5 and intersection / max(min(area_a, area_b), 1) > 0.65


def robot_image_mask(shape, config, intrinsics, arm, q):
    from PIL import Image, ImageDraw

    from ..overlays import project

    image = Image.new("L", (shape[1], shape[0]))
    if arm is None or q is None:
        return np.array(image, bool)
    draw = ImageDraw.Draw(image)
    transform = np.asarray(config["T_base_camera"])
    fx = intrinsics["fx"] if isinstance(intrinsics, dict) else intrinsics.fx
    frames = arm.frames(q)[0]
    for a, b in zip(frames[:-1], frames[1:]):
        pa, pb = project(a, transform, intrinsics), project(b, transform, intrinsics)
        if pa is None or pb is None:
            continue
        camera_z = ((a + b) / 2 - transform[:3, 3]) @ transform[:3, 2]
        radius = int(np.clip(fx * 0.045 / max(camera_z, 0.1), 4, 100))
        draw.line((pa, pb), fill=255, width=radius * 2)
        for x, y in (pa, pb):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    if hasattr(arm, "chain"):
        import cv2

        from ..self_collision import world_parts

        for part in world_parts(arm, q)[0]:
            world = part["vertices"]
            camera = (world - transform[:3, 3]) @ transform[:3, :3]
            camera = camera[camera[:, 2] > 0.05]
            if len(camera) < 4:
                continue

            def val(name):
                return (
                    intrinsics[name]
                    if isinstance(intrinsics, dict)
                    else getattr(intrinsics, name)
                )

            pixels = np.c_[
                camera[:, 0] / camera[:, 2] * val("fx") + val("ppx"),
                camera[:, 1] / camera[:, 2] * val("fy") + val("ppy"),
            ]
            hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
            draw.polygon([tuple(p) for p in hull], fill=255)
    return np.array(image, bool)


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(
        np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0
    ).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
    return float(intersection / max(union, 1))


class YoloDetector:
    def __init__(self, model=None):
        self.model = model
        self.key_model = None
        self.previous_keys = []
        self.previous = []
        self.signature = None
        self.previous_q = None
        self.alignment_failures = 0

    def load(self):
        if self.model is None:
            root = Path("runs/deskbuddy/perception").resolve()
            os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "settings"))
            os.environ["YOLO_AUTOINSTALL"] = "false"
            from ultralytics import YOLOE, settings

            settings.update({"sync": False})
            weights = root / "yoloe-26s-seg-pf.pt"
            if not weights.exists():
                raise FileNotFoundError(f"Install YOLOE-26 Small weights: {weights}")
            self.model = YOLOE(str(weights))

    def detect(self, rgb, depth, intrinsics, config, arm=None, q=None, secondary=None):
        began = time.perf_counter()
        capture_check_time = time.time()
        signature = repr((config.get("T_base_camera"), config.get("test_bed")))
        if signature != self.signature:
            self.previous = []
            self.previous_keys = []
            self.signature = signature
        self.load()
        import torch

        paired = False
        if config.get("task_mode") == "can_discard" and secondary is not None:
            from ..multiview import calibration_current

            second_rgb, stamp = secondary
            paired = capture_check_time - stamp < 1 and calibration_current(
                config, (second_rgb.shape[1], second_rgb.shape[0])
            )
        inputs = np.ascontiguousarray(rgb[..., ::-1])
        if paired:
            inputs = [inputs, np.ascontiguousarray(secondary[0][..., ::-1])]
        predictions = self.model.predict(
            inputs,
            device=0 if torch.cuda.is_available() else "cpu",
            imgsz=640,
            conf=0.28,
            retina_masks=True,
            quantize=16 if torch.cuda.is_available() else 32,
            verbose=False,
            max_det=30,
            agnostic_nms=True,
        )
        result = predictions[0]
        boxes, outlines = [], []
        robot_mask = robot_image_mask(rgb.shape, config, intrinsics, arm, q)
        if q is not None and self.previous_q is not None:
            robot_mask |= robot_image_mask(
                rgb.shape, config, intrinsics, arm, self.previous_q
            )
        self.previous_q = None if q is None else np.array(q).copy()
        second_instances, second_calibration = [], None
        secondary_report = dict(available=secondary is not None, calibrated=False)
        if secondary is not None:
            from ..multiview import calibration_current
            from .multiview_evidence import instances_from_result, secondary_instances

            second_rgb, stamp = secondary
            if capture_check_time - stamp < 1 and calibration_current(
                config, (second_rgb.shape[1], second_rgb.shape[0])
            ):
                second_calibration = config["secondary_camera_calibration"]
                from ..multiview import blue_agreement

                agreement = blue_agreement(
                    rgb,
                    second_rgb,
                    second_calibration["primary_pixels"],
                    second_calibration["secondary_pixels"],
                )
                if agreement is not None:
                    self.alignment_failures = (
                        self.alignment_failures + 1 if agreement < 0.2 else 0
                    )
                secondary_report["blue_agreement"] = agreement
                if self.alignment_failures >= 3:
                    secondary_report["alignment_warning"] = (
                        "Blue markings obscured or alignment changed; review paper corners"
                    )
            if second_calibration is not None:
                second_instances = (
                    instances_from_result(predictions[1])
                    if paired
                    else secondary_instances(
                        self.model,
                        second_rgb,
                        0 if torch.cuda.is_available() else "cpu",
                        None,
                    )
                )
                secondary_report.update(
                    calibrated=True,
                    metric_3d_validated=second_calibration.get(
                        "metric_3d_validated", False
                    ),
                    detections=[
                        {k: v for k, v in item.items() if k != "mask"}
                        for item in second_instances
                    ],
                )
        if config.get("task_mode") == "can_discard":
            from .multiview_evidence import instances_from_result
            from .rgb_can_perception import detect_rgb_cans

            boxes, rgb_report = detect_rgb_cans(
                instances_from_result(result),
                second_instances,
                config,
                second_calibration,
                robot_mask,
                intrinsics,
            )
            secondary_report.update(rgb_report.pop("secondary"))
            return boxes, dict(
                model="YOLOE-26s-seg-pf",
                primary="RGB",
                estimated=True,
                depth_role="display only; unused for can detection or geometry",
                processing_ms=(time.perf_counter() - began) * 1000,
                detection_unix_s=time.time(),
                key_detections=[],
                secondary_view=secondary_report,
                **rgb_report,
            )
        flat_paper, paper, plane_bias, paper_report = paper_surface(
            rgb,
            depth,
            intrinsics,
            config,
            robot_mask,
        )
        paper_rejected = 0
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        key_detections = []
        key_result = result
        key_masks = masks
        key_source = "prompt-free semantic"
        reference = Path("runs/deskbuddy/perception/keys-visual-prompts-s.pt")
        if (
            config.get("task_mode") != "can_discard"
            and reference.exists()
            and model_is_real(self.model)
        ):
            if self.key_model is None:
                from ultralytics import YOLOE

                weights = reference.parent / "yoloe-26s-seg.pt"
                if weights.exists():
                    self.key_model = YOLOE(str(weights))
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.key_model.to(device)
                    cached = torch.load(
                        reference, map_location=device, weights_only=True
                    )
                    self.key_model.set_classes(cached["names"], cached["embeddings"])
            if self.key_model is not None:
                key_result = self.key_model.predict(
                    np.ascontiguousarray(rgb[..., ::-1]),
                    device=0 if torch.cuda.is_available() else "cpu",
                    imgsz=640,
                    conf=0.35,
                    retina_masks=True,
                    verbose=False,
                    max_det=5,
                )[0]
                key_masks = (
                    key_result.masks.data.cpu().numpy()
                    if key_result.masks is not None
                    else []
                )
                key_source = "visual reference"
        for detection, mask in zip(key_result.boxes, key_masks):
            name = key_result.names[int(detection.cls.item())].lower().replace("_", " ")
            if name in (
                "key",
                "keys",
                "keychain",
                "key chain",
                "keyring",
                "key ring",
                "key fob",
                "bunch of keys",
            ):
                roi = [int(v) for v in detection.xyxy[0].cpu().numpy()]
                if (roi[2] - roi[0]) * (roi[3] - roi[1]) > rgb.shape[0] * rgb.shape[
                    1
                ] * 0.15:
                    continue
                valid = (mask > 0.5) & np.isfinite(depth) & (depth > 0.1) & (depth < 2)
                position = None
                if valid.sum() > 5:
                    v, u = np.nonzero(valid)
                    z = float(np.median(depth[valid]))

                    def val(k):
                        return (
                            intrinsics[k]
                            if isinstance(intrinsics, dict)
                            else getattr(intrinsics, k)
                        )

                    camera = np.array(
                        [
                            (np.median(u) - val("ppx")) / val("fx") * z,
                            (np.median(v) - val("ppy")) / val("fy") * z,
                            z,
                        ]
                    )
                    t = np.asarray(config["T_base_camera"])
                    position = (t[:3, :3] @ camera + t[:3, 3]).tolist()
                key_detections.append(
                    dict(
                        label=name,
                        confidence=float(detection.conf.item()),
                        pixel_roi=roi,
                        position_m=position,
                        source=key_source,
                        grasp_pixel=mask_grasp_pixel(mask),
                    )
                )
        for detection, mask in zip(result.boxes, masks):
            roi = tuple(int(v) for v in detection.xyxy[0].cpu().numpy())
            if any(roi_contains_key(roi, key["pixel_roi"]) for key in key_detections):
                continue
            label = result.names[int(detection.cls.item())].lower().replace("_", " ")
            if label in (
                "key",
                "keys",
                "keychain",
                "key chain",
                "keyring",
                "key ring",
                "key fob",
                "bunch of keys",
            ):
                continue
            # Broad scene predictions (room/table) are not movable instances.
            roi_fraction = (
                (roi[2] - roi[0]) * (roi[3] - roi[1]) / (rgb.shape[0] * rgb.shape[1])
            )
            if (mask > 0.5).mean() > 0.65 or roi_fraction > 0.75:
                continue
            instance = mask > 0.5
            original_instance = instance.copy()
            paper_fraction = (instance & paper).sum() / max(instance.sum(), 1)
            flat_fraction = (instance & flat_paper).sum() / max(instance.sum(), 1)
            instance &= ~flat_paper
            paper_like = paper_fraction > 0.5 and flat_fraction > 0.35
            if (instance & robot_mask).sum() / max(instance.sum(), 1) > 0.45:
                continue
            instance &= ~robot_mask
            if any(overlap(roi, item["roi"]) > 0.7 for item in outlines):
                continue
            observed, _ = detect_boxes(
                depth,
                intrinsics,
                config,
                arm,
                q,
                min_pixels=60,
                instance_mask=instance,
                plane_bias_m=plane_bias if paper_report["available"] else None,
            )
            # A paper-sized RGB instance can contain a few noisy edge pixels or
            # a small object over its edge. That does not make the sheet raised.
            raised_fraction = sum(b.pixels for b in observed) / max(
                original_instance.sum(), 1
            )
            if (
                paper_like
                and flat_fraction > 0.7
                and (not observed or max(b.height_m for b in observed) < 0.03)
            ):
                paper_rejected += 1
                continue
            if observed and raised_fraction < 0.15 and not paper_like:
                continue
            if observed:
                box = observed[0]
                # A single RGB instance can have disconnected valid depth patches.
                box.low = np.min([b.low for b in observed], axis=0)
                box.high = np.max([b.high for b in observed], axis=0)
                box.height_m = max(b.height_m for b in observed)
                if observed:
                    # Draw only the raised, in-bounds region retained as geometry.
                    roi = (
                        min(b.pixel_roi[0] for b in observed),
                        min(b.pixel_roi[1] for b in observed),
                        max(b.pixel_roi[2] for b in observed),
                        max(b.pixel_roi[3] for b in observed),
                    )
                box.pixel_roi = roi
                box.source = "YOLOE-26s RGB + filtered depth"
                box.confidence = float(detection.conf.item())
                duplicate = False
                for existing in boxes:
                    intersection = np.maximum(
                        np.minimum(box.high, existing.high)
                        - np.maximum(box.low, existing.low),
                        0,
                    ).prod()
                    smaller = min(
                        (box.high - box.low).prod(),
                        (existing.high - existing.low).prod(),
                    )
                    if intersection / max(smaller, 1e-12) > 0.65:
                        duplicate = True
                        break
                if duplicate:
                    continue
                boxes.append(box)
                outlines.append(dict(roi=roi, geometry="estimated"))
            else:
                matched = [
                    b
                    for b in self.previous
                    if overlap(roi, b[1].pixel_roi) > 0.6
                    or roi_contains_key(roi, b[1].pixel_roi)
                ]
                if matched and time.monotonic() - matched[0][0] < 2:
                    box = copy.deepcopy(matched[0][1])
                    box.pixel_roi = roi
                    boxes.append(box)
                    outlines.append(dict(roi=roi, geometry="cached"))
        if second_calibration is not None:
            from .multiview_evidence import (
                confirm_box,
                filter_secondary,
                fuse_fragments,
                raised_cloud,
            )

            cloud, primary_pixels = raised_cloud(
                depth,
                intrinsics,
                config,
                robot_mask,
                plane_bias if paper_report["available"] else None,
            )
            second_instances, projected = filter_secondary(
                second_instances, second_calibration, config, cloud
            )
            accepted, accepted_outlines, votes = [], [], []
            for box, outline in zip(boxes, outlines):
                confirmed, reason = confirm_box(
                    box, second_instances, second_calibration
                )
                votes.append(
                    dict(roi=box.pixel_roi, confirmed=confirmed, reason=reason)
                )
                if confirmed:
                    accepted.append(box)
                    accepted_outlines.append(outline)
            if projected is not None:
                boxes = fuse_fragments(
                    accepted,
                    second_instances,
                    second_calibration,
                    cloud,
                    primary_pixels,
                    projected,
                )
                outlines = [dict(roi=b.pixel_roi, geometry="estimated") for b in boxes]
            else:
                boxes, outlines = accepted, accepted_outlines
            secondary_report.update(
                votes=votes,
                detections=[
                    {k: v for k, v in item.items() if k != "mask"}
                    for item in second_instances
                ],
                fusion="paired RGB masks + measured raised points"
                if projected is not None
                else "table-contact confirmation only",
                geometry_estimated=not second_calibration.get(
                    "metric_3d_validated", False
                ),
                foreground_points=len(cloud),
            )
        elif config.get("secondary_camera"):
            boxes, outlines = [], []
            secondary_report["fusion"] = (
                "Awaiting fresh calibrated second view; no new one-view obstacles"
            )
        key_history = []
        for key in key_detections:
            matched = next(
                (
                    old
                    for old in self.previous_keys
                    if key_overlap(key["pixel_roi"], old["pixel_roi"]) > 0.5
                ),
                None,
            )
            key["confirmed_frames"] = (
                matched.get("confirmed_frames", 1) + 1 if matched else 1
            )
            observed = list(key["pixel_roi"])
            history = (
                (matched or {}).get("_history", []) + [observed + key["grasp_pixel"]]
            )[-5:]
            key["observed_pixel_roi"] = observed
            median = [int(v) for v in np.rint(np.median(history, axis=0))]
            key["pixel_roi"], key["grasp_pixel"] = median[:4], median[4:]
            saved = copy.deepcopy(key)
            saved["_history"] = history
            key_history.append(saved)
        self.previous_keys = key_history
        # Do not refresh timestamps for cached geometry: a hole has a bounded lifetime.
        now = time.monotonic()
        recent = [(t, b) for t, b in self.previous if now - t < 2]
        fresh = [b for b, o in zip(boxes, outlines) if o["geometry"] == "estimated"]
        self.previous = [(now, b) for b in fresh] + [
            (t, b)
            for t, b in recent
            if not any(overlap(b.pixel_roi, new.pixel_roi) > 0.6 for new in fresh)
        ]
        return boxes, dict(
            model="YOLOE-26s-seg-pf",
            primary="RGB",
            outlines=outlines,
            processing_ms=(time.perf_counter() - began) * 1000,
            estimated=True,
            paper_surface=dict(paper_report, rejected_instances=paper_rejected),
            key_detections=key_detections,
            detection_unix_s=time.time(),
            secondary_view=secondary_report,
        )


class YoloService:
    """One outstanding frame, no queue growth or model work on Tk's thread."""

    def __init__(self):
        self.detector = YoloDetector()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self.future = None
        self.signature = None
        self.tracker = ObjectTracker(min_hits=2, max_age_s=10, max_misses=40)
        self.stable = StableBoxes(
            confirmations=2, removal_confirmations=3, world_m=0.012, pixels=10
        )
        self.last_image = None
        self.last_secondary_image = None
        self.last_inference = 0.0
        self.skipped_frames = 0
        # Raised by the GUI while a path preview is being computed, so object
        # detection geometry does not compete with planning for the CPU.
        self.min_interval_s = 0.05

    def update(self, rgb, depth, intrinsics, config, arm=None, q=None, secondary=None):
        signature = repr(
            tuple(
                config.get(k)
                for k in (
                    "T_base_camera",
                    "test_bed",
                    "workspace_min_m",
                    "workspace_max_m",
                    "calibration_verified",
                    "tool_offset_m",
                    "secondary_camera_calibration",
                )
            )
        )
        output = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if self.signature == signature:
                    boxes, report = result
                    report["raw_obstacle_boxes"] = [b.as_dict() for b in boxes]
                    boxes = self.stable.update(self.tracker.update(boxes))
                    report.update(
                        scene_revision=self.stable.revision,
                        change_pending=self.stable.pending,
                        skipped_frames=self.skipped_frames,
                        tracking="low-motion Kalman + Hungarian association",
                        stability=dict(
                            world_deadband_m=0.012,
                            pixel_deadband=10,
                            confirmations=2,
                            removal_confirmations=3,
                            birth_observations=2,
                            occlusion_retention_s=10,
                        ),
                    )
                    report["outlines"] = [
                        dict(roi=b.pixel_roi, geometry="stable") for b in boxes
                    ]
                    output = boxes, report
            except Exception as exc:
                output = ([], {"model": "YOLOE-26s-seg-pf", "error": str(exc)})
            self.future = None
        if self.future is None:
            if self.signature != signature:
                self.tracker = ObjectTracker(min_hits=2, max_age_s=10, max_misses=40)
                self.stable = StableBoxes(
                    confirmations=2, removal_confirmations=3, world_m=0.012, pixels=10
                )
                self.last_image = None
            now = time.monotonic()
            image = image_signature(rgb)
            second_image = (
                image_signature(secondary[0]) if secondary is not None else None
            )
            secondary_changed = second_image is not None and image_changed(
                self.last_secondary_image, second_image
            )
            elapsed = now - self.last_inference
            # Periodic refresh detects slow changes and updates depth estimates.
            if elapsed < getattr(self, "min_interval_s", 0.05) or (
                elapsed < 2.0
                and not self.stable.pending
                and not self.tracker.pending
                and not image_changed(self.last_image, image)
                and not secondary_changed
            ):
                self.skipped_frames += 1
                return output
            self.signature = signature
            self.last_image = image
            self.last_secondary_image = second_image
            self.last_inference = now
            self.future = self.executor.submit(
                self.detector.detect,
                rgb.copy(),
                depth.copy(),
                intrinsics,
                copy.deepcopy(config),
                arm,
                None if q is None else np.array(q).copy(),
                **(
                    {"secondary": (secondary[0].copy(), secondary[1])}
                    if secondary is not None
                    else {}
                ),
            )
        return output

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
