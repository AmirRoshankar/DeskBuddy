import copy
import json

import numpy as np
from test_scene_stability import box

from deskbuddy.trajectory.object_tracker import ObjectTracker


def test_kalman_tracker_reduces_box_jitter_and_limits_each_update():
    clock = [0.0]
    tracker = ObjectTracker(lambda: clock[0])
    tracker.update([box()])
    rng = np.random.default_rng(42)
    raw = []
    filtered = []
    previous = 0.12
    for _ in range(100):
        clock[0] += 0.25
        observed = box()
        noise = rng.normal(0, 0.008, 3)
        observed.low += noise
        observed.high += noise
        estimate = tracker.update([observed])[0]
        raw.append((observed.low + observed.high) / 2)
        filtered.append((estimate.low + estimate.high) / 2)
        assert abs(filtered[-1][0] - previous) <= 0.00401
        previous = filtered[-1][0]
    assert np.std(filtered[20:], axis=0).mean() < np.std(raw[20:], axis=0).mean() * 0.6
    json.dumps(estimate.as_dict())


def test_association_survives_reordering_and_transient_jump():
    clock = [0.0]
    tracker = ObjectTracker(lambda: clock[0])
    initial = tracker.update([box(), box(0.3)])
    ids = [b.track_id for b in initial]
    clock[0] += 0.3
    reordered = tracker.update([box(0.3), box()])
    assert [b.track_id for b in reordered] == ids
    jumped = copy.deepcopy(box())
    jumped.low += 0.08
    jumped.high += 0.08
    clock[0] += 0.3
    tracker.update([jumped, box(0.3)])
    assert tracker.tracks[0].outlier_count == 1
    assert np.allclose(tracker.tracks[0].estimate().low, initial[0].low)
    clock[0] += 0.3
    recovered = tracker.update([box(), box(0.3)])
    assert np.allclose(recovered[0].low, initial[0].low, atol=0.004)
