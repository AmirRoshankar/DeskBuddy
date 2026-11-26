"""Held-out synthetic comparison. No motor/camera imports or hardware actions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..kinematics import Arm
from ..planning import plan_reach
from .data import check_budget
from .inference import Proposal


def summary(rows):
    valid = [r for r in rows if r["success"]]
    errors = [r["gap_mm"] for r in rows if "gap_mm" in r]
    return dict(
        samples=len(rows),
        valid_joint_solution_rate=sum(r.get("within_limits", False) for r in rows)
        / len(rows),
        success_at_2mm_rate=len(valid) / len(rows),
        median_model_gap_mm=float(np.median(errors)) if errors else None,
        p95_model_gap_mm=float(np.percentile(errors, 95)) if errors else None,
        median_ms=float(np.median([r["ms"] for r in rows])),
        p95_ms=float(np.percentile([r["ms"] for r in rows], 95)),
        path_checked=sum("path_valid" in r for r in rows),
        valid_checked_paths=sum(r.get("path_valid", False) for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/flowreach")
    parser.add_argument("--samples", type=int, default=600)
    args = parser.parse_args()
    folder = Path(args.folder)
    check_budget(folder, 5_000_000)
    meta = json.loads((folder / "dataset.json").read_text())
    config = meta["config"]
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    source = np.load(folder / "dataset.npz")
    inputs = source["test_x"][: args.samples].astype(float)
    proposals = {
        name: Proposal(folder / (name + ".pt"), config, name) for name in ("mlp", "imf")
    }
    for model in proposals.values():
        for _ in range(10):
            model.predict(inputs[0, :5], inputs[0, 5:], check_domain=False)
    reports = {
        name: []
        for name in ("numerical", "mlp_raw", "imf_raw", "mlp_refined", "imf_refined")
    }
    for index, x in enumerate(inputs):
        start, target = x[:5], x[5:]

        def score(q, elapsed):
            gap = float(np.linalg.norm(arm.fk(q) - target) * 1000)
            within = bool(((q >= arm.lower) & (q <= arm.upper)).all())
            return dict(
                gap_mm=gap,
                within_limits=within,
                success=within and gap <= config["ik_tolerance_m"] * 1000,
                ms=elapsed,
            )

        begun = time.perf_counter()
        try:
            q = arm.ik(target, start, config["ik_tolerance_m"], config["locked_joints"])
            row = score(q, (time.perf_counter() - begun) * 1000)
        except ValueError as exc:
            row = dict(
                success=False, ms=(time.perf_counter() - begun) * 1000, error=str(exc)
            )
        reports["numerical"].append(row)
        for name, model in proposals.items():
            begun = time.perf_counter()
            q = model.predict(start, target, check_domain=False)
            prediction_ms = (time.perf_counter() - begun) * 1000
            raw = score(q, prediction_ms)
            reports[name + "_raw"].append(raw)
            # This table tests warm-start quality, separately from deployment's
            # 30 mm gate/domain guard and entire-path validation below.
            try:
                q_refined = arm.ik(
                    target,
                    start,
                    config["ik_tolerance_m"],
                    config["locked_joints"],
                    initial_guess=q,
                )
                row = score(q_refined, (time.perf_counter() - begun) * 1000)
            except ValueError as exc:
                row = dict(
                    success=False,
                    ms=(time.perf_counter() - begun) * 1000,
                    error=str(exc),
                )
            reports[name + "_refined"].append(row)
        if index < 100:
            for name, mode in (
                ("numerical", "numerical"),
                ("imf_refined", "flow_assisted"),
            ):
                try:
                    plan = plan_reach(
                        arm,
                        start,
                        target,
                        dict(
                            config,
                            planner_mode=mode,
                            flow_checkpoint=str(folder / "imf.pt"),
                            control_hz=20,
                        ),
                    )
                    reports[name][-1].update(
                        path_valid=True, planner_info=plan.planner_info
                    )
                except ValueError:
                    reports[name][-1]["path_valid"] = False
        if (index + 1) % 100 == 0:
            print(f"Evaluated {index + 1}/{len(inputs)} held-out targets", flush=True)
    result = dict(
        dataset_signature=meta["signature"],
        split=meta["split_method"],
        synthetic_only=True,
        hardware_tested=False,
        imf_nfe=1,
        noise_seed=42,
        latency_device="CPU; loaded/warmed models; excludes model loading",
        metrics={k: summary(v) for k, v in reports.items()},
        rows=reports,
    )
    payload = json.dumps(result, indent=2)
    check_budget(folder, len(payload.encode()))
    (folder / "benchmark.json").write_text(payload, encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
