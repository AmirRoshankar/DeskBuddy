"""Diagnostic camera/TCP fit with fixed joint calibration; never enables motion."""

import hashlib
import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import fit_rigid, transform_point
from .kinematics import Arm


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def fit_tool_point(rows):
    if len(rows) != 8 or len({r["session_id"] for r in rows}) != 1:
        raise ValueError("Need eight poses from one session")
    camera = np.array([r["camera_point_m"] for r in rows], float)
    frames = np.array([Arm().frames(r["q_model_rad"])[1] for r in rows])
    base = frames[:, :3, 3]
    t, _ = fit_rigid(camera[:6], base[:6])
    initial = np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3], np.zeros(3)]

    def residual(x):
        observed = camera @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:6]
        modeled = base + np.einsum("nij,j->ni", frames[:, :3, :3], x[6:9])
        return observed - modeled

    fit = least_squares(
        lambda x: residual(x)[:6].ravel(),
        initial,
        bounds=(np.r_[[-np.inf] * 6, [-0.025] * 3], np.r_[[np.inf] * 6, [0.025] * 3]),
    )
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if (
        not fit.success
        or singular[-1] < 1e-6
        or singular[0] / singular[-1] > 1e5
        or np.linalg.norm(fit.x[6:]) >= 0.025
    ):
        raise ValueError(
            "Tool offset is too large or not adequately constrained by these poses"
        )
    errors = np.linalg.norm(residual(fit.x), axis=1) * 1000
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    t[:3, 3] = fit.x[3:6]
    return {
        "session_id": rows[0]["session_id"],
        "camera_serial": rows[0]["camera_serial"],
        "created_unix_s": time.time(),
        "training_sha256": fingerprint(rows),
        "T_base_camera": t.tolist(),
        "tool_offset_m": fit.x[6:].tolist(),
        "fit_error_mm": errors[:6].tolist(),
        "heldout_error_mm": errors[6:].tolist(),
        "passed_initial_checks": bool(errors.max() <= 12.0),
        "validated": False,
        "joint_offsets_changed": False,
        "note": "Candidate clicked jaw-tip location relative to the URDF tool frame. Needs two fresh poses; no active config or motor changes.",
    }


def check_tool_point(candidate, rows):
    if fingerprint(rows[:8]) != candidate["training_sha256"]:
        raise ValueError("Original observations changed after fitting")
    checks = rows[8:]
    if len(checks) < 2:
        raise ValueError("Need two fresh check poses")
    errors = []
    for row in checks:
        if (
            row["session_id"] != candidate["session_id"]
            or row["unix_time_s"] <= candidate["created_unix_s"]
        ):
            raise ValueError(
                "Checks must be fresh poses from the same fixed-camera session"
            )
        frame = Arm().frames(row["q_model_rad"])[1]
        tip = frame[:3, 3] + frame[:3, :3] @ np.array(candidate["tool_offset_m"])
        errors.append(
            float(
                np.linalg.norm(
                    transform_point(candidate["T_base_camera"], row["camera_point_m"])
                    - tip
                )
                * 1000
            )
        )
    return {
        "fresh_check_error_mm": errors,
        "passed": bool(candidate["passed_initial_checks"] and max(errors) <= 12.0),
        "refitted_on_checks": False,
        "joint_offsets_changed": False,
    }
