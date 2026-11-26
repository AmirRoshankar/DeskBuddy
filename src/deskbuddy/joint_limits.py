"""Model-angle bounds derived from the saved, hand-measured encoder ranges."""

import numpy as np

from .kinematics import JOINTS


def calibrated_limits(calibration, signs, offsets_deg):
    signs, offsets = np.asarray(signs, float), np.asarray(offsets_deg, float)
    if (
        signs.shape != (5,)
        or offsets.shape != (5,)
        or not np.isfinite(offsets).all()
        or not np.isin(signs, [-1, 1]).all()
    ):
        raise ValueError("Need five joint signs and finite model zero offsets")
    result = np.empty((2, 5))
    for i, name in enumerate(JOINTS):
        low, high = (calibration[name][key] for key in ("range_min", "range_max"))
        if not 0 <= low < high <= 4095:
            raise ValueError(f"Invalid saved encoder range for {name}")
        # LeRobot DEGREES normalization uses this midpoint and 4095 counts/rev.
        midpoint = (low + high) / 2
        endpoints = (np.array([low, high]) - midpoint) * 360 / 4095
        result[:, i] = np.sort(endpoints * signs[i] + offsets[i])
    return result.tolist()
