import cv2
import numpy as np

from deskbuddy.point_tracking import PointTracker


def test_tracks_clicked_nonorange_feature_and_ignores_orange_distractor():
    first = np.full((160, 240, 3), 60, np.uint8)
    cv2.rectangle(first, (70, 60), (100, 95), (230, 230, 230), -1)
    cv2.rectangle(first, (130, 60), (160, 95), (255, 100, 0), -1)
    tracker = PointTracker(first, [71, 61], 1.0)
    second = cv2.warpAffine(first, np.float32([[1, 0, 7], [0, 1, 4]]), (240, 160))
    result = tracker.update(second, 2.0)
    assert result["valid"]
    assert np.linalg.norm(np.asarray(result["pixel"]) - [78, 65]) < 1
    assert tracker.update(first, 1.0) == result


def test_tracker_does_not_invent_measurement_when_feature_disappears():
    first = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(first, (40, 40), (60, 60), (240, 240, 240), -1)
    tracker = PointTracker(first, [40, 40], 1.0)
    result = tracker.update(np.zeros_like(first), 2.0)
    assert not result["valid"]
    assert result["unix_time_s"] == 1.0
