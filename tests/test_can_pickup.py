import numpy as np
import pytest

from deskbuddy.can_discard import checked_transfer
from deskbuddy.can_pickup import match_motion


def test_rgb_tracker_distinguishes_stationary_lifted_and_ambiguous():
    rng = np.random.default_rng(12)
    patch = rng.integers(0, 256, (30, 22, 3), dtype=np.uint8)
    original = np.array([80, 100])
    expected = np.array([80, 65])
    image = np.zeros((160, 160, 3), np.uint8)
    image[85:115, 69:91] = patch
    assert match_motion(image, patch, original, expected) is False
    image[:] = 0
    image[50:80, 69:91] = patch
    assert match_motion(image, patch, original, expected) is True
    assert match_motion(image, patch, original, original + [0, 3]) is None
    assert match_motion(image * 0, patch, original, expected) is None


def drive(decisions, stop=lambda: False):
    events = []
    outcomes = iter(decisions)
    result = checked_transfer(
        lambda attempt: events.append(("attempt", attempt)) or (55, 2),
        lambda index: events.append(("move", index)),
        lambda value: events.append(("grip", value)),
        lambda: next(outcomes),
        stop,
        lambda report: None,
    )
    return result, events


def test_failed_lift_retries_new_attempt_and_only_transfers_after_evidence():
    result, events = drive([False, False, True])
    assert result["pickup_verified"] and result["can_attempts"] == 3
    assert [e for e in events if e[0] == "attempt"] == [
        ("attempt", 0),
        ("attempt", 1),
        ("attempt", 2),
    ]
    assert events.count(("move", 3)) == 1
    assert events[-1] == ("grip", 55)


def test_five_failed_lifts_never_transfer_or_drop():
    result, events = drive([False] * 5)
    assert not result["completed"] and not result["pickup_verified"]
    assert result["can_attempts"] == 5
    assert ("move", 3) not in events
    assert events[-1] == ("move", 2)


def test_uncertain_evidence_or_stop_never_retries_or_opens_after_lift():
    with pytest.raises(RuntimeError, match="uncertain"):
        drive([None])
    with pytest.raises(RuntimeError, match="stopped"):
        drive([], stop=lambda: True)


def test_raw_snapshot_images_are_fresh_and_have_no_annotation_dependency(tmp_path):
    import time

    from deskbuddy.can_pickup import read_images
    from deskbuddy.live_snapshot import pickup_images

    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[..., 1] = 200
    report = pickup_images(tmp_path, rgb, (rgb, time.time()))
    report.update(intrinsics={}, camera_stale=False)
    stamp, images, _ = read_images(lambda: report)
    assert len(images) == 2 and images[0].shape == rgb.shape
    assert np.mean(images[0][..., 1]) > 190
    assert read_images(lambda: report, newer_than=stamp) is None
    report["camera_stale"] = True
    assert read_images(lambda: report) is None
