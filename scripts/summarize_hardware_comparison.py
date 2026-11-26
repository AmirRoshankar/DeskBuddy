"""Export the small hardware comparison without publishing camera images."""

import json
from pathlib import Path

from deskbuddy.power_estimate import QuasiStaticPower
from deskbuddy.storage import load_plan


def main():
    root = Path("runs/hardware-comparison/2026-11-25")
    trials = []
    for name in ("imf-up", "mlp-return", "mlp-up"):
        trial = json.loads((root / name / "trial.json").read_text(encoding="utf-8"))
        job = Path(trial["job_directory"])
        plan, payload = load_plan(job / "executed_plan.json")
        power = QuasiStaticPower(payload["config"]).evaluate(plan)
        trial["trial"] = name
        trial["planned_motion_power_estimate"] = {
            key: power[key]
            for key in ("energy_j", "mean_power_w", "peak_power_w", "duration_s")
        }
        trials.append(trial)
    report = {
        "date": "2026-11-25",
        "protocol": "1 cm vertical reaches, iMF up, MLP return, MLP up, attempted iMF return; each accepted proposal refined with numerical IK before motion",
        "trials": trials,
        "rejected_trials": [
            {
                "trial": "imf-return",
                "executed": False,
                "reason": "Learned proposal outside calibrated joint ranges; numerical fallback excluded from comparison",
            }
        ],
        "limitations": [
            "Endpoint gaps are FK estimates from joint feedback, not independent physical tip measurements.",
            "Only three executed reaches; starts differ slightly, and the planned counterbalanced comparison was incomplete. No general model ranking is justified.",
            "The UI green marker remained the previous clicked target, not the external comparison target.",
            "Power is a quasi-static scenario for the 2-second planned path only, excluding settling and subsequent holding; electrical current was not measured. See flowreach-analysis.md for assumptions.",
            "First iMF and MLP-return reached model tolerance but timed out on the stricter joint settling criterion; MLP-up met both.",
        ],
    }
    Path("docs/research/results/flowreach-hardware.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Small hardware comparison — 25 November 2026",
        "",
        report["protocol"] + ".",
        "",
        "| Trial | Model gap (mm) | Peak tracking error (deg) | Elapsed (s) | Loop (Hz) | Joint settling met | Planned energy estimate (J) |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for t in trials:
        c = t["control_timing"]
        lines.append(
            f"| {t['trial']} | {t['model_gap_mm']:.2f} | "
            f"{max(t['max_tracking_error_deg']):.2f} | {t['duration_s']:.2f} | "
            f"{c['achieved_hz']:.1f} | {c['settling_converged']} | "
            f"{t['planned_motion_power_estimate']['energy_j']:.2f} |"
        )
    lines += [
        "",
        "All three executed reaches completed within the configured model tolerance, with zero read/write failures or goal mismatches. The final iMF return was rejected before motor movement because its raw prediction exceeded calibrated joint bounds.",
        "",
    ]
    lines += ["- " + item for item in report["limitations"]]
    lines += [
        "",
        "The last verified state was hold enabled at the raised position, with gripper torque off. After testing, viewer position reads began timing out on all motor IDs; current hold could no longer be verified. No torque-release command was sent.",
        "",
        "[Detailed results](flowreach-hardware.json). Raw trial logs and camera snapshots remain local under `runs/hardware-comparison/2026-11-25/`.",
        "",
    ]
    Path("docs/research/flowreach-hardware.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
