"""SORT-inspired association and low-motion Kalman tracking for desk boxes."""

import copy
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def measurement(box):
    return np.r_[(box.low + box.high) / 2, box.high - box.low, box.pixel_roi]


class BoxTrack:
    def __init__(self, box, identifier, now):
        self.box = copy.deepcopy(box)
        self.identifier = identifier
        self.x = np.r_[measurement(box), np.zeros(10)]
        self.p = np.diag(
            np.r_[
                np.full(6, 0.01**2),
                np.full(4, 4.0**2),
                np.full(6, 0.005**2),
                np.full(4, 2.0**2),
            ]
        )
        self.last_seen = self.last_update = now
        self.misses = 0
        self.outlier = None
        self.outlier_count = 0
        self.hits = 1

    def update(self, observed, now):
        self.hits += 1
        dt = float(np.clip(now - self.last_update, 0.05, 2.0))
        transition = np.eye(20)
        transition[:10, 10:] = np.eye(10) * dt
        self.x[10:] *= np.exp(-dt * 2)
        predicted = transition @ self.x
        covariance = transition @ self.p @ transition.T + np.diag(
            np.r_[
                np.full(6, 0.001**2),
                np.full(4, 0.5**2),
                np.full(6, 0.002**2),
                np.full(4, 1.0**2),
            ]
            * dt
        )
        value = measurement(observed)
        residual = value - predicted[:10]
        noise = np.diag(
            np.r_[np.full(3, 0.008**2), np.full(3, 0.012**2), np.full(4, 4.0**2)]
        )
        innovation = covariance[:10, :10] + noise
        # Gate jumps in world position, dimensions and image coordinates together.
        distance = float(residual @ np.linalg.solve(innovation, residual))
        if distance > 45:
            if (
                self.outlier is not None
                and np.max(np.abs(value[:6] - self.outlier[:6])) < 0.012
                and np.max(np.abs(value[6:] - self.outlier[6:])) < 10
            ):
                self.outlier_count += 1
            else:
                self.outlier, self.outlier_count = value, 1
            if self.outlier_count < 3:
                self.last_seen = now
                return
            # Confirmed relocation: reinitialize, never invent a slow swept obstacle.
            self.__init__(observed, self.identifier, now)
            return
        self.outlier, self.outlier_count = None, 0
        gain = np.linalg.solve(innovation, covariance[:, :10].T).T
        corrected = predicted + gain @ residual
        limits = np.r_[np.full(3, 0.004), np.full(3, 0.003), np.full(4, 3.0)]
        corrected[:10] = self.x[:10] + np.clip(
            corrected[:10] - self.x[:10], -limits, limits
        )
        self.x = corrected
        # Joseph form preserves covariance symmetry/positive semidefiniteness.
        kh = np.zeros((20, 20))
        kh[:, :10] = gain
        self.p = (np.eye(20) - kh) @ covariance @ (
            np.eye(20) - kh
        ).T + gain @ noise @ gain.T
        self.last_seen = self.last_update = now
        self.misses = 0
        self.box = copy.deepcopy(observed)

    def estimate(self):
        box = copy.deepcopy(self.box)
        center, size = self.x[:3], np.maximum(self.x[3:6], 0.004)
        box.low, box.high = center - size / 2, center + size / 2
        box.pixel_roi = tuple(int(v) for v in np.rint(self.x[6:10]))
        box.height_m = float(size[2])
        box.track_id = self.identifier
        return box


class ObjectTracker:
    def __init__(self, clock=time.monotonic, min_hits=1, max_age_s=2.5, max_misses=4):
        self.clock = clock
        self.min_hits, self.max_age_s, self.max_misses = min_hits, max_age_s, max_misses
        self.tracks = []
        self.next_id = 1

    @property
    def pending(self):
        return any(
            t.misses or t.outlier_count or t.hits < self.min_hits for t in self.tracks
        )

    def update(self, boxes):
        now = self.clock()
        costs = np.full((len(self.tracks), len(boxes)), 1e6)
        for i, track in enumerate(self.tracks):
            for j, box in enumerate(boxes):
                delta = np.linalg.norm(measurement(box)[:3] - track.x[:3])
                if delta < 0.15:
                    costs[i, j] = (
                        delta
                        + np.linalg.norm(measurement(box)[3:6] - track.x[3:6]) * 0.2
                    )
        matched_tracks, matched_boxes = set(), set()
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                self.tracks[i].update(boxes[j], now)
                matched_tracks.add(i)
                matched_boxes.add(j)
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.misses += 1
        self.tracks = [
            t
            for t in self.tracks
            if t.misses < self.max_misses and now - t.last_seen < self.max_age_s
        ]
        for j, box in enumerate(boxes):
            if j not in matched_boxes:
                self.tracks.append(BoxTrack(box, self.next_id, now))
                self.next_id += 1
        return [t.estimate() for t in self.tracks if t.hits >= self.min_hits]
