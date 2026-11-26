"""Fit an external camera from hand-moved, stationary tool-point observations.

The clicked physical point must coincide with the URDF tool frame. Fits are
provisional until held-out errors and physical model conventions are checked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .geometry import fit_rigid, transform_point


def fit_observations(records, max_error_m=0.012):
    if len(records) < 8:
        raise ValueError(
            "Need eight distinct tool observations: six fit points and two held-out points"
        )
    if (
        len({r["session_id"] for r in records}) != 1
        or len({r["camera_serial"] for r in records}) != 1
    ):
        raise ValueError("Do not mix camera sessions or cameras")
    camera = np.array([r["camera_point_m"] for r in records], float)
    base = np.array([r["base_point_model_m"] for r in records], float)
    if (
        camera.shape != (len(records), 3)
        or base.shape != camera.shape
        or not np.isfinite(camera).all()
        or not np.isfinite(base).all()
    ):
        raise ValueError("Invalid point arrays")
    train_count = len(records) - 2
    # A volume of observations also tests height-dependent/model errors.
    singular = np.linalg.svd(
        base[:train_count] - base[:train_count].mean(0), compute_uv=False
    )
    if (
        singular[-1] < 0.012
        or np.linalg.norm(np.ptp(base[:train_count], axis=0)) < 0.10
    ):
        raise ValueError("Fit points need more spread, including different heights")
    t, train_error = fit_rigid(camera[:train_count], base[:train_count])
    heldout_error = np.array(
        [
            np.linalg.norm(transform_point(t, p) - b)
            for p, b in zip(camera[train_count:], base[train_count:])
        ]
    )
    report = {
        "fit_points": train_count,
        "heldout_points": 2,
        "fit_rmse_mm": float(np.sqrt(np.mean(train_error**2)) * 1000),
        "fit_max_error_mm": float(train_error.max() * 1000),
        "heldout_error_mm": (heldout_error * 1000).tolist(),
        "threshold_mm": max_error_m * 1000,
        "passed": bool(max(train_error.max(), heldout_error.max()) <= max_error_m),
        "T_base_camera": t.tolist(),
        "camera_serial": records[0]["camera_serial"],
        "session_id": records[0]["session_id"],
        "note": "Errors compare camera observations with model-derived tool positions, not independent metrology.",
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="runs/tool_points.jsonl")
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--output", default="runs/tool_calibration_report.json")
    p.add_argument("--max-error-mm", type=float, default=12.0)
    args = p.parse_args()
    config = load_config(args.config)
    if not np.isfinite(args.max_error_mm) or args.max_error_mm <= 0:
        p.error("Error threshold must be positive")
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        p.error("No tool observations")
    # Reopening the viewer starts a fresh session; don't silently mix old poses.
    rows = [r for r in rows if r["session_id"] == rows[-1]["session_id"]]
    report = fit_observations(rows, args.max_error_mm / 1000)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    if config["camera_serial"] != report["camera_serial"]:
        p.error("Observation camera does not match configured camera")
    config["T_base_camera"] = report["T_base_camera"]
    config["calibration_verified"] = False
    Path(args.config).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        "Saved provisional transform. Model mapping and physical tool point still require review before execution."
    )


if __name__ == "__main__":
    main()
