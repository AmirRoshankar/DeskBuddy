"""Track a user-selected image feature without any colour classification."""

import cv2
import numpy as np


class PointTracker:
    def __init__(self, image, pixel, stamp):
        self.gray = cv2.cvtColor(np.asarray(image).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        self.pixel = np.asarray(pixel, np.float32)
        self.stamp = float(stamp)
        self.valid = True

    def update(self, image, stamp):
        if stamp <= self.stamp:
            return self.result()
        if not self.valid:
            return self.result()
        gray = cv2.cvtColor(np.asarray(image).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if gray.shape != self.gray.shape:
            self.valid = False
            return self.result()
        options = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        )
        start = self.pixel.reshape(1, 1, 2)
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, start, None, **options
        )
        if forward is None or not bool(ok[0, 0]):
            self.valid = False
            return self.result()
        backward, ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, forward, None, **options
        )
        p = forward.ravel()
        if (
            backward is None
            or not bool(ok[0, 0])
            or np.linalg.norm(backward - start) > 1.5
            or not (0 <= p[0] < gray.shape[1] and 0 <= p[1] < gray.shape[0])
        ):
            self.valid = False
            return self.result()
        self.gray, self.pixel, self.stamp = gray, p.copy(), float(stamp)
        return self.result()

    def result(self):
        return dict(
            pixel=self.pixel.tolist(),
            unix_time_s=self.stamp,
            valid=self.valid,
            feature="fixed jaw fingertip",
            source="user-selected optical flow",
        )
