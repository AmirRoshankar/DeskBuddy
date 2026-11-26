"""Estimate small model-only joint-zero corrections, then validate on NEW poses.

This never writes motor registers or updates the active configuration. Base-pan
and wrist-roll offsets remain fixed; the first is confounded with camera rotation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import fit_rigid, transform_point
from .kinematics import Arm


def fingerprint(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit_candidate(rows):
    if len(rows) < 8 or len({r["session_id"] for r in rows}) != 1:
        raise ValueError(
            "Need eight or more observations from one fixed-camera session"
        )
    arm = Arm()
    camera = np.array([r["camera_point_m"] for r in rows])
    q = np.array([r["q_model_rad"] for r in rows])
    if (
        camera.shape != (len(rows), 3)
        or q.shape != (len(rows), 5)
        or not np.isfinite(camera).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Malformed observations")
    base = np.array([arm.fk(pose) for pose in q])
    t, _ = fit_rigid(camera, base)
    initial = np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3], np.zeros(3)]

    def residual(x):
        observed = camera @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
        corrected = np.array([arm.fk(pose + np.r_[0, x[6:9], 0]) for pose in q])
        return (observed - corrected).ravel()

    bound = np.deg2rad(10.0)
    result = least_squares(
        residual,
        initial,
        bounds=(np.r_[[-np.inf] * 6, [-bound] * 3], np.r_[[np.inf] * 6, [bound] * 3]),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=300,
    )
    if not result.success or np.abs(result.x[6:]).max() > bound * 0.99:
        raise ValueError(
            "Correction fit failed or hit the 10-degree bound; inspect the model and observations"
        )
    singular = np.linalg.svd(result.jac, compute_uv=False)
    if singular[-1] <= 1e-6 or singular[0] / singular[-1] > 1e5:
        raise ValueError("These poses do not constrain the correction well enough")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t[:3, 3] = result.x[3:6]
    errors = np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1) * 1000
    return {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "session_id": rows[0]["session_id"],
        "training_count": len(rows),
        "training_sha256": fingerprint(rows),
        "camera_serial": rows[0]["camera_serial"],
        "T_base_camera": t.tolist(),
        "joint_offset_correction_deg": np.rad2deg(np.r_[0, result.x[6:9], 0]).tolist(),
        "training_error_mm": errors.tolist(),
        "training_rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "training_max_error_mm": float(errors.max()),
        "jacobian_condition": float(singular[0] / singular[-1]),
        "validated": False,
        "note": "Exploratory model correction. Requires fresh held-out captures; motor calibration is unchanged.",
    }


def validate_candidate(candidate, rows, threshold_m=0.012):
    count = candidate["training_count"]
    if fingerprint(rows[:count]) != candidate["training_sha256"]:
        raise ValueError("Training observations changed after the candidate was frozen")
    checks = rows[count:]
    if len(checks) < 2:
        raise ValueError(
            "Capture at least two NEW check poses after the candidate was frozen"
        )
    if any(
        r["session_id"] != candidate["session_id"]
        or r["camera_serial"] != candidate["camera_serial"]
        or r["unix_time_s"] <= candidate["created_unix_s"]
        for r in checks
    ):
        raise ValueError(
            "Check poses must be fresh captures from the same unchanged camera session"
        )
    arm = Arm()
    correction = np.deg2rad(candidate["joint_offset_correction_deg"])
    errors = []
    for row in checks:
        observed = transform_point(candidate["T_base_camera"], row["camera_point_m"])
        modeled = arm.fk(np.array(row["q_model_rad"]) + correction)
        errors.append(float(np.linalg.norm(observed - modeled)))
    return {
        "session_id": candidate["session_id"],
        "check_count": len(checks),
        "new_check_error_mm": (np.array(errors) * 1000).tolist(),
        "threshold_mm": threshold_m * 1000,
        "passed": bool(max(errors) <= threshold_m),
        "refitted_on_checks": False,
        "note": "Tests the frozen correction on fresh observations; not independent external metrology.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "validate"))
    p.add_argument("--input", default="runs/tool_points.jsonl")
    p.add_argument("--candidate", default="runs/model_calibration_candidate.json")
    p.add_argument("--report", default="runs/model_calibration_validation.json")
    args = p.parse_args()
    rows = [
        json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()
    ]
    if not rows:
        p.error("No observations")
    rows = [r for r in rows if r["session_id"] == rows[-1]["session_id"]]
    if args.mode == "fit":
        result = fit_candidate(rows)
        path = Path(args.candidate)
    else:
        result = validate_candidate(json.loads(Path(args.candidate).read_text()), rows)
        path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.mode == "validate" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
