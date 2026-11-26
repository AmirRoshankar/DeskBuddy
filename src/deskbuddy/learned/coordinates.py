"""Workspace remapping for candidate IK seeds, not for physical execution goals."""

import numpy as np


def map_workspace_target(target, current, training):
    low, high = (
        np.asarray(current["workspace_min_m"]),
        np.asarray(current["workspace_max_m"]),
    )
    train_low, train_high = (
        np.asarray(training["workspace_min_m"]),
        np.asarray(training["workspace_max_m"]),
    )
    point = np.asarray(target, float)
    if (
        point.shape != (3,)
        or not np.isfinite([point, low, high, train_low, train_high]).all()
        or (high <= low).any()
        or (train_high <= train_low).any()
    ):
        raise ValueError("Invalid workspace remapping")
    return train_low + (point - low) / (high - low) * (train_high - train_low)
