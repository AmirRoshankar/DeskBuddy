import copy
from concurrent.futures import Future

import numpy as np

from deskbuddy.trajectory.perception import DetectedBox
from deskbuddy.trajectory.scene_stability import (
    StableBoxes,
    image_changed,
    image_signature,
)
from deskbuddy.trajectory.yolo_perception import YoloService


def box(x=0.1):
    return DetectedBox(
        np.array([x, 0.1, 0]),
        np.array([x + 0.04, 0.14, 0.08]),
        100,
        0.08,
        (40, 40, 80, 80),
    )


def test_jitter_order_and_confidence_do_not_change_published_scene():
    stable = StableBoxes()
    objects = [box(), box(0.25)]
    assert stable.update(objects) == []
    stable.update(objects)
    assert stable.revision == 1
    noisy = copy.deepcopy(objects[::-1])
    for b in noisy:
        b.low += 0.003
        b.high += 0.003
        b.pixel_roi = (42, 42, 82, 82)
        b.confidence = 0.9
    for _ in range(20):
        result = stable.update(noisy)
    assert stable.revision == 1
    assert np.array_equal(result[0].low, objects[0].low)


def test_real_motion_cumulative_drift_and_removal_are_confirmed():
    stable = StableBoxes()
    for _ in range(2):
        stable.update([box()])
    stable.update([box(0.104)])
    stable.update([box(0.110)])
    assert stable.revision == 1
    stable.update([box(0.111)])
    assert stable.revision == 2
    stable.update([])
    stable.update([box(0.111)])  # transient drop is forgotten
    assert not stable.pending
    stable.update([])
    stable.update([])
    assert len(stable.boxes) == 1
    assert stable.update([]) == []


def test_blurred_image_gate_ignores_noise_and_exposure_but_detects_object():
    rng = np.random.default_rng(42)
    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    reference = image_signature(image)
    noisy = np.clip(
        image.astype(float) + rng.normal(0, 4, image.shape) + 10, 0, 255
    ).astype(np.uint8)
    assert not image_changed(reference, image_signature(noisy))
    image[150:250, 200:300] = 200
    assert image_changed(reference, image_signature(image))


def test_service_skips_static_image_but_refreshes_and_confirms_changes(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(
        "deskbuddy.trajectory.yolo_perception.time.monotonic", lambda: now[0]
    )
    service = YoloService()
    service.executor.shutdown()
    calls = []

    class ImmediateExecutor:
        def submit(self, *args):
            calls.append(args)
            future = Future()
            future.set_result(([box()], {}))
            return future

    service.executor = ImmediateExecutor()
    rgb = np.full((100, 100, 3), 100, dtype=np.uint8)
    args = (rgb, np.ones((100, 100)), {}, {})
    service.update(*args)
    result = None
    for _ in range(6):
        now[0] += 0.3
        result = service.update(*args) or result
    # Two births plus two confirmations, then the static image is skipped.
    assert len(calls) == 3 and len(result[0]) == 1
    now[0] += 0.3
    service.update(*args)
    assert len(calls) == 3
    now[0] += 2
    service.update(*args)
    assert len(calls) == 4
