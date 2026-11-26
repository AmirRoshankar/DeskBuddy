"""Offline feasibility/timing experiment for execute-coarse-then-refine ordering."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..kinematics import Arm
from ..planning import Plan, interpolate_joints, plan_reach, validate_plan
from .data import check_budget
from .inference import Proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/flowreach")
    args = parser.parse_args()
    folder = Path(args.folder)
    meta = json.loads((folder / "dataset.json").read_text())
    config = dict(meta["config"], planner_mode="numerical", control_hz=20)
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    model = Proposal(folder / "imf.pt", config)
    inputs = np.load(folder / "dataset.npz")["test_x"][:100].astype(float)
    rows = []
    for x in inputs:
        start, target = x[:5], x[5:]
        proposed = model.predict(start, target, check_domain=False)
        gap = float(np.linalg.norm(arm.fk(proposed) - target))
        if gap > 0.01:
            continue
        try:
            baseline = plan_reach(arm, start, target, config)
            times, poses = interpolate_joints(start, proposed, config)
            coarse = validate_plan(
                arm, Plan(times, poses, arm.fk(proposed), 0.0), config
            )
            # An ideal simulated endpoint stands in for a fresh encoder reading.
            # No physical bus/controller is opened and no real accuracy is inferred.
            correction = plan_reach(arm, coarse.joints_rad[-1], target, config)
        except ValueError:
            continue
        rows.append(
            dict(
                baseline_s=float(baseline.time_s[-1]),
                coarse_s=float(coarse.time_s[-1]),
                correction_s=float(correction.time_s[-1]),
                raw_gap_mm=gap * 1000,
                corrected_gap_mm=correction.endpoint_error_m * 1000,
            )
        )
    result = dict(
        synthetic_only=True,
        hardware_tested=False,
        attempted=len(inputs),
        both_stages_valid=len(rows),
        interpolation_speed_deg_s=config["max_joint_speed_deg_s"],
        ideal_encoder_endpoint=True,
        excludes_settling_and_reacquisition_overhead=True,
        rows=rows,
    )
    if rows:
        result.update(
            median_extra_trajectory_s=float(
                np.median(
                    [r["coarse_s"] + r["correction_s"] - r["baseline_s"] for r in rows]
                )
            ),
            median_correction_trajectory_s=float(
                np.median([r["correction_s"] for r in rows])
            ),
            max_corrected_model_gap_mm=max(r["corrected_gap_mm"] for r in rows),
        )
    payload = json.dumps(result, indent=2)
    check_budget(folder, len(payload.encode()))
    (folder / "coarse_refinement.json").write_text(payload, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
