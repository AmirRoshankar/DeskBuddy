import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .geometry import validate_transform

DEFAULT = {
    "workspace_min_m": [0.05, -0.30, 0.04],
    "workspace_max_m": [0.45, 0.30, 0.45],
    "table_z_m": 0.0,
    "assume_level_table": False,
    "link_clearance_m": 0.015,
    "hover_m": 0.06,
    "ik_tolerance_m": 0.002,
    "goal_tolerance_m": 0.01,
    "max_joint_speed_deg_s": 24.0,
    "control_hz": 30,
    "preview_start_deg": [0, -15, 30, 15, 0],
    "T_base_camera": None,
    "calibration_verified": False,
    "camera_serial": None,
    "secondary_camera": None,
    "secondary_camera_calibration": None,
    "robot_port": None,
    "robot_id": "deskbuddy",
    "motor_calibration_dir": "runs/motor-calibration",
    "joint_mapping_verified": False,
    "joint_signs": [1, 1, 1, 1, 1],
    "joint_offsets_deg": [0, 0, 0, 0, 0],
    "tool_offset_m": [0, 0, 0],
    "test_bed": None,
    "locked_joints": ["wrist_roll"],
    "model_joint_limits_deg": None,
    "validated_wrist_roll_deg": None,
    "wrist_roll_tolerance_deg": 5.0,
    "planner_mode": "numerical",
    "trajectory_method": "rrt",
    "trajectory_candidates": 1,
    "allow_object_collisions": False,
    "click_and_grab": False,
    "grasp_request": None,
    "home_request": False,
    "can_request": None,
    "discard_zone": None,
    "task_mode": "reach",
    "flow_checkpoint": "runs/flowreach/imf.pt",
    "proposal_method": "imf",
    "flow_seed_strategy": "direct",
    "restrict_to_test_bed": True,
    "show_workspace": True,
}


def load_config(path=None):
    c = deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(c)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        c.update(supplied)
    for key in (
        "workspace_min_m",
        "workspace_max_m",
        "preview_start_deg",
        "joint_signs",
        "joint_offsets_deg",
        "tool_offset_m",
    ):
        size = 3 if key.startswith("workspace") or key == "tool_offset_m" else 5
        a = np.asarray(c[key], float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key}")
    if np.linalg.norm(c["tool_offset_m"]) > 0.025:
        raise ValueError("Tool offset exceeds the supported 25 mm calibration range")
    if not np.all(np.array(c["workspace_max_m"]) > c["workspace_min_m"]):
        raise ValueError("Workspace bounds reversed")
    if not np.isin(c["joint_signs"], [-1, 1]).all():
        raise ValueError("Joint signs must be +1 or -1")
    for key in (
        "hover_m",
        "ik_tolerance_m",
        "goal_tolerance_m",
        "max_joint_speed_deg_s",
        "control_hz",
        "link_clearance_m",
        "wrist_roll_tolerance_deg",
    ):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not np.isfinite(c["table_z_m"]):
        raise ValueError("Invalid table height")
    if c["control_hz"] > 500 or c["max_joint_speed_deg_s"] > 30:
        raise ValueError("This demo supports at most 500 Hz requested and 30 degrees/s")
    if c["T_base_camera"] is not None:
        validate_transform(c["T_base_camera"])
    if c["test_bed"] is not None:
        from .test_bed import corners

        corners(c)
    if not isinstance(c["locked_joints"], list) or set(c["locked_joints"]) - {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    }:
        raise ValueError("Unknown locked joints")
    if c["model_joint_limits_deg"] is not None:
        bounds = np.asarray(c["model_joint_limits_deg"], float)
        if (
            bounds.shape != (2, 5)
            or not np.isfinite(bounds).all()
            or (bounds[0] >= bounds[1]).any()
        ):
            raise ValueError("Invalid measured model joint limits")
    if c["validated_wrist_roll_deg"] is not None and not np.isfinite(
        c["validated_wrist_roll_deg"]
    ):
        raise ValueError("Invalid wrist-roll reference")
    for key in (
        "assume_level_table",
        "calibration_verified",
        "joint_mapping_verified",
        "restrict_to_test_bed",
        "show_workspace",
        "allow_object_collisions",
        "click_and_grab",
        "home_request",
    ):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if c["task_mode"] not in ("reach", "can_discard"):
        raise ValueError("Unknown task mode")
    if c["discard_zone"] is not None:
        zone = np.asarray(c["discard_zone"], float)
        if zone.shape != (4, 3) or not np.isfinite(zone).all():
            raise ValueError("Discard zone requires four XYZ corners")
    if c["can_request"] is not None and not isinstance(c["can_request"], dict):
        raise ValueError("Invalid can request")
    if c["planner_mode"] not in ("numerical", "flow_assisted"):
        raise ValueError("Unknown planner mode")
    if c["trajectory_method"] not in ("direct", "rrt", "mlp", "fm", "mf", "imf"):
        raise ValueError("Unknown trajectory planner")
    if type(c["trajectory_candidates"]) is not int or c[
        "trajectory_candidates"
    ] not in (1, 4):
        raise ValueError("Use one or four trajectory candidates")
    if c["proposal_method"] not in ("imf", "mlp"):
        raise ValueError("Unknown learned proposal method")
    if c["flow_seed_strategy"] not in ("direct", "workspace_affine", "best"):
        raise ValueError("Unknown flow seed strategy")
    if not isinstance(c["flow_checkpoint"], str) or not c["flow_checkpoint"]:
        raise ValueError("Flow checkpoint must be a nonempty path")
    request = c["grasp_request"]
    if request is not None:
        if not isinstance(request, dict):
            raise ValueError("grasp_request must be an object or null")
        unknown = set(request) - {"source_m", "max_retries", "reference_pixel"}
        if unknown:
            raise ValueError(f"Unknown grasp_request keys: {sorted(unknown)}")
        source = np.asarray(request.get("source_m"), float)
        if source.shape != (3,) or not np.isfinite(source).all():
            raise ValueError("grasp_request.source_m must be three finite coordinates")
        retries = request.get("max_retries", 5)
        if type(retries) is not int or not 0 <= retries <= 5:
            raise ValueError("grasp_request.max_retries must be an integer from 0 to 5")
        if request.get("reference_pixel") is not None:
            pixel = np.asarray(request["reference_pixel"], float)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError(
                    "grasp_request.reference_pixel must be two finite coordinates or null"
                )
    if c["secondary_camera"] is not None:
        camera = c["secondary_camera"]
        if not isinstance(camera, dict) or not isinstance(camera.get("name", ""), str):
            raise ValueError("Invalid secondary camera configuration")
    if c["secondary_camera_calibration"] is not None and not isinstance(
        c["secondary_camera_calibration"], dict
    ):
        raise ValueError("Invalid secondary camera calibration")
    if c["secondary_camera_calibration"] is not None:
        calibration = c["secondary_camera_calibration"]
        h = np.asarray(calibration.get("H_secondary_paper"), float)
        points = np.asarray(calibration.get("corners_base_m"), float)
        if (
            h.shape != (3, 3)
            or not np.isfinite(h).all()
            or abs(np.linalg.det(h)) < 1e-12
            or points.shape != (4, 3)
            or not np.isfinite(points).all()
        ):
            raise ValueError("Invalid A4 camera alignment geometry")
        if calibration.get("T_base_camera") is not None:
            validate_transform(calibration["T_base_camera"])
    return c
