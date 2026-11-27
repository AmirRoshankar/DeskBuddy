"""Matched-case, offline motor-power scenarios for the three endpoint solvers."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..kinematics import Arm
from ..planning import plan_reach
from ..power_estimate import MotorAssumptions, QuasiStaticPower
from .data import check_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/deskbuddy")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--voltage", type=float, default=5.4)
    args = parser.parse_args()
    folder = Path(args.folder)
    meta = json.loads((folder / "dataset.json").read_text())
    config = dict(meta["config"], control_hz=20)
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    estimator = QuasiStaticPower(config, MotorAssumptions(voltage_v=args.voltage))
    x = np.load(folder / "dataset.npz")["test_x"][: args.samples].astype(float)
    rows, examples = [], {}
    for index, sample in enumerate(x):
        start, target = sample[:5], sample[5:]
        plans = {}
        try:
            for name in ("numerical", "mlp", "imf"):
                plans[name] = plan_reach(
                    arm,
                    start,
                    target,
                    dict(
                        config,
                        planner_mode="numerical"
                        if name == "numerical"
                        else "flow_assisted",
                        flow_checkpoint=str(folder / (name + ".pt")),
                        _proposal_method=name,
                    ),
                )
        except ValueError:
            continue  # Compare exactly the same valid requests for every method.
        record = dict(case=index)
        for name, plan in plans.items():
            estimate = estimator.evaluate(plan)
            if len(rows) == 0:
                examples[name] = estimate
            record[name] = {
                k: estimate[k]
                for k in (
                    "energy_j",
                    "energy_wh",
                    "mean_power_w",
                    "peak_power_w",
                    "duration_s",
                    "per_joint_energy_j",
                )
            }
            record[name]["planner_info"] = plan.planner_info
        rows.append(record)
        if len(rows) % 10 == 0:
            print(f"Estimated {len(rows)} matched valid cases", flush=True)
    summary = {}
    for name in ("numerical", "mlp", "imf"):
        summary[name] = (
            {
                "median_" + key: float(np.median([r[name][key] for r in rows]))
                for key in (
                    "energy_j",
                    "energy_wh",
                    "mean_power_w",
                    "peak_power_w",
                    "duration_s",
                )
            }
            if rows
            else {}
        )
    result = dict(
        estimated=True,
        measured=False,
        motor_io=False,
        requested_cases=len(x),
        matched_valid_cases=len(rows),
        voltage_v=args.voltage,
        active_motors=5,
        gripper_motor_off=True,
        summary=summary,
        assumptions=examples.get("numerical", {}).get("assumptions"),
        limitations=examples.get("numerical", {}).get("limitations"),
        source_urls=[
            "https://docs.waveshare.com/ST3215_Servo",
            "https://www.dfrobot.com/product-2961.html",
        ],
        rows=rows,
        example_power_curves=examples,
    )
    payload = json.dumps(result, indent=2)
    check_budget(folder, len(payload.encode()))
    (folder / "energy_comparison.json").write_text(payload, encoding="utf-8")
    print(
        json.dumps({k: result[k] for k in ("matched_valid_cases", "summary")}, indent=2)
    )


if __name__ == "__main__":
    main()
