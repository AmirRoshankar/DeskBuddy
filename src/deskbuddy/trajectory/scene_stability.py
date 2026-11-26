"""Cheap RGB change gate and hysteresis for published obstacle geometry."""

import copy

import numpy as np
from PIL import Image, ImageFilter


def image_signature(rgb):
    return np.asarray(
        Image.fromarray(rgb)
        .convert("L")
        .resize((160, 120))
        .filter(ImageFilter.GaussianBlur(1.2)),
        dtype=float,
    )


def image_changed(previous, current):
    if previous is None:
        return True
    delta = current - previous
    # Ignore uniform exposure shifts; retain local object motion/appearance.
    delta -= np.median(delta)
    return float((np.abs(delta) > 18).mean()) >= 0.025


def match_boxes(previous, current, world_m=0.008, pixels=8):
    """Match independent of detector ordering; confidence is not scene geometry."""
    if len(previous) != len(current):
        return None
    remaining = list(current)
    ordered = []
    for old in previous:
        if not remaining:
            return None
        costs = [
            np.max(np.abs(np.r_[b.low, b.high] - np.r_[old.low, old.high]))
            for b in remaining
        ]
        index = int(np.argmin(costs))
        candidate = remaining.pop(index)
        if (
            costs[index] > world_m
            or np.max(np.abs(np.array(candidate.pixel_roi) - old.pixel_roi)) > pixels
        ):
            return None
        ordered.append(candidate)
    return ordered


class StableBoxes:
    def __init__(
        self, confirmations=2, removal_confirmations=3, world_m=0.008, pixels=8
    ):
        self.world_m, self.pixels = world_m, pixels
        self.boxes = []
        self.candidate = None
        self.count = 0
        self.revision = 0
        self.confirmations, self.removal_confirmations = (
            confirmations,
            removal_confirmations,
        )

    @property
    def pending(self):
        return self.candidate is not None

    def update(self, observed):
        if match_boxes(self.boxes, observed, self.world_m, self.pixels) is not None:
            self.candidate = None
            self.count = 0
            return copy.deepcopy(self.boxes)
        if (
            self.candidate is not None
            and match_boxes(self.candidate, observed, self.world_m, self.pixels)
            is not None
        ):
            self.count += 1
        else:
            self.candidate = copy.deepcopy(observed)
            self.count = 1
        # Disappearances need extra confirmation to survive short occlusions.
        required = (
            self.removal_confirmations
            if len(observed) < len(self.boxes)
            else self.confirmations
        )
        if self.count >= required:
            self.boxes = copy.deepcopy(observed)
            self.candidate = None
            self.count = 0
            self.revision += 1
        return copy.deepcopy(self.boxes)
