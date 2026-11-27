"""Camera-only UI interaction checks with synthetic pixels; no devices."""

import sys
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from deskbuddy.ui.camera_alignment import CameraAlignment


def test_drag_updates_source_pixels_and_save_accepts_warning():
    root = tk.Tk()
    root.withdraw()
    saved = []
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    points = [[100, 100], [400, 100], [400, 400], [100, 400]]
    config = {
        "test_bed": {"corner_pixels": points},
        "T_base_camera": np.eye(4).tolist(),
    }
    with patch(
        "deskbuddy.ui.camera_alignment.paper_candidate", return_value=np.array(points)
    ):
        view = CameraAlignment(
            root, image, image, np.zeros((480, 640)), {}, config, saved.append
        )
    root.update()
    try:
        label = view.labels[0]
        x, y = label.viewport.from_camera(100, 100)
        view.click(0, SimpleNamespace(x=x + 1, y=y + 1))
        x, y = label.viewport.from_camera(150, 180)
        event = SimpleNamespace(x=x + 1, y=y + 1)
        view.drag(0, event)
        view.release(0, event)
        assert np.linalg.norm(np.array(view.points[0][0]) - [150, 180]) < 5
        assert view.dragging is None
        assert str(view.save["state"]) == "normal"
        texts = [
            child.cget("text")
            for frame in view.window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button)
        ]
        assert not any("Rotate" in text or "Reverse" in text for text in texts)
        view.result = {
            "validation_warnings": ["Depth differs"],
            "metric_3d_validated": False,
        }
        view.commit()
        assert saved and saved[0]["validation_warnings"]
    finally:
        if not view.closed:
            view.close()
        root.destroy()


def test_unlimited_matches_camera_only_saved_without_depth_arrays():
    import json

    view = CameraAlignment.__new__(CameraAlignment)
    view.result = {"camera_only": True, "metric_3d_validated": False}
    view.status = SimpleNamespace(set=lambda message: None)
    view.tip_observations = [
        {
            "primary_pixel": [i, i + 1],
            "secondary_pixel": [i + 2, i + 3],
            "depth": np.zeros((4, 4)),
        }
        for i in range(12)
    ]
    view.update_tip_checks()
    assert len(view.result["matched_points"]) == 12
    assert not view.result["metric_3d_validated"]
    json.dumps(view.result)


def test_manual_matches_validate_and_revoke_on_failed_check():
    view = CameraAlignment.__new__(CameraAlignment)
    view.result = {"metric_3d_validated": False}
    view.status = SimpleNamespace(set=lambda message: None)
    view.config, view.intrinsics = {}, {}
    view.tip_observations = [
        {"primary_pixel": [i, i], "secondary_pixel": [i, i]} for i in range(5)
    ]
    with patch(
        "deskbuddy.ui.camera_alignment.check_elevated_tip",
        return_value={"passed": True, "error_px": [1.0, 2.0]},
    ):
        view.update_tip_checks()
    assert view.result["metric_3d_validated"]
    with patch(
        "deskbuddy.ui.camera_alignment.check_elevated_tip",
        side_effect=ValueError("Missing depth"),
    ):
        view.update_tip_checks()
    assert not view.result["metric_3d_validated"]


def test_alignment_import_with_already_loaded_multiview(monkeypatch):
    import importlib

    from deskbuddy import multiview
    from deskbuddy.ui import camera_alignment

    # The running UI may have imported multiview before the new helper existed.
    monkeypatch.delattr(multiview, "check_elevated_tip", raising=False)
    importlib.reload(camera_alignment)
    assert callable(camera_alignment.check_elevated_tip)


def test_open_without_depth_and_edit_does_not_mutate_config():
    root = tk.Tk()
    root.withdraw()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    points = [[100, 100], [400, 100], [400, 400], [100, 400]]
    config = {"test_bed": {"corner_pixels": points}}
    with patch(
        "deskbuddy.ui.camera_alignment.paper_candidate", return_value=np.array(points)
    ):
        view = CameraAlignment(
            root, image, image, None, {}, config, lambda result: None
        )
    try:
        root.update()
        view.points[0][0][0] = 150
        assert config["test_bed"]["corner_pixels"][0][0] == 100
        assert view.window.winfo_exists()
        view.result = {"camera_only": True}
        view.begin_tip()
        assert view.tip_mode
    finally:
        view.close()
        root.destroy()


def test_live_feed_point_capture_before_check_and_window_controls():
    root = tk.Tk()
    root.withdraw()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 0.5)
    counter = [0]

    def frames():
        counter[0] += 1
        return np.full_like(image, counter[0] % 255), image.copy(), depth.copy()

    with patch("deskbuddy.ui.camera_alignment.paper_candidate", return_value=None):
        view = CameraAlignment(
            root,
            image,
            image,
            depth,
            {},
            {},
            lambda result: None,
            frames_provider=frames,
        )
    root.update()
    try:
        assert view.result is None
        assert not view.window.transient()
        assert view.window.resizable() == (1, 1)
        assert view.window.bind("<F11>")
        before = counter[0]
        view.refresh_live()
        assert counter[0] > before
        assert int(view.images[0][0, 0, 0]) == counter[0]
        corners_before = [list(points) for points in view.points]
        view.begin_tip()
        for i in range(2):
            x, y = view.labels[i].viewport.from_camera(200, 200)
            view.click(i, SimpleNamespace(x=x, y=y))
        assert len(view.tip_observations) == 1
        assert view.points == corners_before
        assert "A4 pose not checked" in view.status.get()
        assert np.all(view.tip_observations[0]["depth"] == 0.5)
        texts = [
            child.cget("text")
            for frame in view.window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button)
        ]
        assert not any("Reset" in text or "snapshot" in text.lower() for text in texts)
        view.begin_tip()
        assert view.tip_mode
    finally:
        view.close()
        root.destroy()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows native maximize/fullscreen state"
)
def test_fullscreen_button_enters_and_exits_native_fullscreen():
    root = tk.Tk()
    root.withdraw()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("deskbuddy.ui.camera_alignment.paper_candidate", return_value=None):
        view = CameraAlignment(root, image, image, None, {}, {}, lambda result: None)
    try:
        root.update()
        assert view.window.state() == "zoomed"
        view.fullscreen_button.invoke()
        root.update()
        assert bool(view.window.attributes("-fullscreen"))
        assert view.fullscreen_button.cget("text") == "Exit fullscreen"
        view.fullscreen_button.invoke()
        root.update()
        assert not bool(view.window.attributes("-fullscreen"))
        assert view.window.state() == "zoomed"
    finally:
        view.close()
        root.destroy()


def test_save_reopen_restores_exact_latest_alignment_without_detector():
    import json

    root = tk.Tk()
    root.withdraw()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    first = [[100, 100], [400, 100], [400, 400], [100, 400]]
    second = [[110, 120], [410, 120], [410, 420], [110, 420]]
    latest = dict(
        primary_pixels=first,
        secondary_pixels=second,
        camera_only=True,
        calibrated_unix_s=20,
        metric_3d_validated=False,
        matched_points=[dict(primary_pixel=[200, 200], secondary_pixel=[210, 220])],
    )
    config = dict(
        secondary_camera_calibration=dict(
            latest, calibrated_unix_s=10, secondary_pixels=first
        ),
        camera_pair_alignment=latest,
    )
    saved = []
    with patch("deskbuddy.ui.camera_alignment.paper_candidate") as detector:
        view = CameraAlignment(root, image, image, None, {}, config, saved.append)
        root.update()
        assert not detector.called
        assert view.points == [first, second]
        assert all(view.result[key] == value for key, value in latest.items())
        view.points[1][0][0] = 111
        assert config["camera_pair_alignment"]["secondary_pixels"][0][0] == 110
        view.points = [first, second]
        view.commit()
    persisted = json.loads(json.dumps(saved[0]))
    with patch("deskbuddy.ui.camera_alignment.paper_candidate") as detector:
        reopened = CameraAlignment(
            root,
            image,
            image,
            None,
            {},
            dict(camera_pair_alignment=persisted),
            lambda r: None,
        )
        try:
            assert not detector.called
            assert reopened.points == [first, second]
            assert len(reopened.tip_observations) == 1
        finally:
            reopened.close()
            root.destroy()
