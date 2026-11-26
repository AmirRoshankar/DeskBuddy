import json
from pathlib import Path

import numpy as np

from deskbuddy.calibration_session import (
    assess_captures,
    invalidate_camera,
    remember_session,
    restore_session,
)
from deskbuddy.config import load_config


def rows():
    camera = np.random.default_rng(4).uniform(
        [-0.1, -0.1, 0.35], [0.1, 0.1, 0.55], (8, 3)
    )
    return [
        {
            "session_id": "test",
            "camera_serial": "cam",
            "camera_point_m": p.tolist(),
            "base_point_model_m": (p + [0.1, 0.2, 0.3]).tolist(),
        }
        for p in camera
    ]


def test_eighth_capture_automatically_assesses_without_refitting_extra_clicks(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    points = rows()
    assert assess_captures(points[:7]) is None
    report, message = assess_captures(points)
    assert (
        report["passed"]
        and "8 points captured" in message
        and "Go remains disabled" in message
    )
    assert Path("runs/calibration/test/report.json").exists()
    extra = dict(points[-1], base_point_model_m=[99, 99, 99])
    assert assess_captures(points + [extra])[0] == report


def test_reset_invalidates_persisted_transform_and_preserves_old_one(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    config.update(
        calibration_verified=True, T_base_camera=np.eye(4).tolist(), camera_serial="cam"
    )
    invalidate_camera(config, "configs/local.json")
    assert not config["calibration_verified"] and config["T_base_camera"] is None
    assert not json.loads(Path("configs/local.json").read_text())[
        "calibration_verified"
    ]
    assert json.loads(Path("runs/calibration_before_reset.json").read_text())[
        "calibration_verified"
    ]
    remember_session("test", config)
    Path("runs/tool_points.jsonl").write_text("\n".join(json.dumps(r) for r in rows()))
    assert len(restore_session(config)[1]) == 8
    config["camera_serial"] = "other"
    assert restore_session(config) == (None, [])
