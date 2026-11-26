"""Offline IK/path assessment for a visible, raised grid. Never commands hardware."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .geometry import transform_point
from .kinematics import Arm
from .planning import plan_reach
from .test_bed import table_height_at


def assess_grid(config, seed, intrinsics, rows=3, columns=3):
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    inverse = np.linalg.inv(np.asarray(config["T_base_camera"]))
    lo, hi = (
        np.asarray(config["workspace_min_m"]),
        np.asarray(config["workspace_max_m"]),
    )
    xs = np.linspace(lo[0] + 0.005, hi[0] - 0.005, columns)
    ys = np.linspace(lo[1] + 0.005, hi[1] - 0.005, rows)
    current = np.asarray(seed)
    results = []

    def pixel(point):
        p = transform_point(inverse, point)
        if p[2] <= 0.05:
            return None
        return [
            float(intrinsics["fx"] * p[0] / p[2] + intrinsics["ppx"]),
            float(intrinsics["fy"] * p[1] / p[2] + intrinsics["ppy"]),
        ]

    for row, y in enumerate(ys):
        for x in xs if row % 2 == 0 else xs[::-1]:
            surface_z = float(table_height_at([x, y], config))
            z = surface_z + config["hover_m"]
            target = np.array([x, y, z])
            tool_pixel = pixel(target)
            table_pixel = pixel([x, y, surface_z])
            visible = all(
                p is not None and 30 <= p[0] < 610 and 30 <= p[1] < 450
                for p in (tool_pixel, table_pixel)
            )
            record = {
                "target_m": target.tolist(),
                "table_pixel": table_pixel,
                "tool_pixel": tool_pixel,
                "fully_visible": visible,
                "valid": False,
            }
            try:
                plan = plan_reach(arm, current, target, config)
                record.update(
                    valid=visible,
                    duration_s=float(plan.time_s[-1]),
                    model_endpoint_error_mm=plan.endpoint_error_m * 1000,
                )
                current = plan.joints_rad[-1]
                if not visible:
                    record["reason"] = (
                        "Table or tool target is outside the camera margin"
                    )
            except ValueError as exc:
                record["reason"] = str(exc)
            results.append(record)
    return {
        "physical_execution": False,
        "rows": rows,
        "columns": columns,
        "valid_points": sum(r["valid"] for r in results),
        "total_points": len(results),
        "target_height_base_m": z,
        "points": results,
        "note": "Model, joint/path and camera-view checks only. Does not establish physical reachability, tracking accuracy, or obstacle clearance.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--output", default="runs/grid_preview.json")
    args = p.parse_args()
    config = load_config(args.config)
    state = json.loads(Path("runs/live/state.json").read_text())
    motor = state["motor_state"]
    report = assess_grid(
        config, motor.get("q_goal_model_rad", motor["q_model_rad"]), state["intrinsics"]
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "points"}, indent=2))


if __name__ == "__main__":
    main()
