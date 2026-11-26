"""Camera-only UI interaction checks with synthetic pixels; no devices."""

import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
