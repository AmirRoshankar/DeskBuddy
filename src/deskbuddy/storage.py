import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .planning import Plan


def save_plan(path, plan, config, synthetic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "planner": next(
            (
                i["trajectory_method"]
                for i in plan.planner_info or []
                if "trajectory_method" in i
            ),
            "quintic_joint_baseline",
        ),
        "time_s": plan.time_s.tolist(),
        "joints_rad": plan.joints_rad.tolist(),
        "target_m": plan.target_m.tolist(),
        "endpoint_error_m": plan.endpoint_error_m,
        "planner_info": plan.planner_info,
        "recovery_waypoint_m": plan.recovery_waypoint_m.tolist()
        if plan.recovery_waypoint_m is not None
        else None,
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_plan(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unknown plan schema")
    plan = Plan(
        np.array(data["time_s"]),
        np.array(data["joints_rad"]),
        np.array(data["target_m"]),
        float(data["endpoint_error_m"]),
        np.array(data["recovery_waypoint_m"])
        if data.get("recovery_waypoint_m") is not None
        else None,
        data.get("planner_info"),
    )
    return plan, data
