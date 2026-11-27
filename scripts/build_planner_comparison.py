"""Build the main planner table; --reproduce reruns released checkpoints on CPU."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runs/deskbuddy/trajectories-v1"


def bootstrap_comparison(folder=ARTIFACTS / "models-v3", dataset=ARTIFACTS):
    data = np.load(folder / "evaluation-paths.npz")
    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    labels = data["labels"].tolist()
    families = np.array(
        [metadata["scenarios"][i]["family"] for i in data["scenario_index"]]
    )
    unique = np.unique(families)
    draws = np.random.default_rng(20261125).integers(
        0, len(unique), (10000, len(unique))
    )
    result = {}
    for candidates in (1, 4):
        a, b = (
            labels.index(f"{method}-1nfe-{candidates}c") for method in ("imf", "mf")
        )
        difference = data["valid"][a].astype(float) - data["valid"][b].astype(float)
        grouped = np.array([difference[families == family].mean() for family in unique])
        result[str(candidates)] = dict(
            imf_minus_mf_pp=float(difference.mean() * 100),
            family_bootstrap_95ci_pp=np.percentile(
                grouped[draws].mean(axis=1) * 100, [2.5, 97.5]
            ).tolist(),
            layout_families=len(unique),
            resamples=10000,
        )
    return result


def render(source, pairs):
    rows = [
        ("RRT-Connect", "rrt-0nfe-1c"),
        ("MLP", "mlp-1nfe-1c"),
        ("FM, 8 evaluations", "fm-8nfe-1c"),
        ("MF, 1 evaluation", "mf-1nfe-1c"),
        ("iMF, 1 evaluation", "imf-1nfe-1c"),
    ]
    metrics = {m["label"]: m for m in source["metrics"]}
    lines = [
        "# Planner comparison",
        "",
        "500 held-out planning problems from 20 layout families; 245 have a blocked direct route.",
        "",
        "| Planner | Own path valid | Blocked-route valid | Median total ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, key in rows:
        m = metrics[key]
        lines.append(
            f"| {name} | {m['raw_valid_rate']:.1%} | {m['blocked_raw_valid_rate']:.1%} | {m['median_total_ms']:.2f} |"
        )
    multi = metrics["imf-1nfe-4c"]
    interval = pairs["1"]["family_bootstrap_95ci_pp"]
    lines += [
        "",
        f"With four candidates, iMF has at least one valid path in {multi['raw_valid_rate']:.1%} of problems ({multi['blocked_raw_valid_rate']:.1%} with blocked direct routes).",
        "",
        f"The paired layout-family bootstrap estimates iMF minus MF at {pairs['1']['imf_minus_mf_pp']:.1f} percentage points, 95% interval [{interval[0]:.1f}, {interval[1]:.1f}]. This interval conditions on the fixed training checkpoints and sampling seed; training-seed variation is reported separately.",
        "",
        "## What valid means",
        "",
        "Raw validity checks every segment of the complete 32-point XY path against the workspace and inflated obstacle geometry, before RRT fallback. It describes a tool proxy, not full-arm feasibility. Four-candidate validity means at least one candidate passes the same checks.",
        "",
        "Timings are warmed CPU measurements with one Torch thread. Total time includes generation, validation and any RRT fallback; camera processing, joint-space lifting and motor execution are excluded.",
        "",
        "## Interpretation",
        "",
        "iMF is the strongest one-candidate neural result in this run. RRT solves all proxy problems. The harder full-arm experiment exposes a different limit: iMF solves 0/31 blocked cases. The next step is whole-body conditioning and online replanning, not assuming the proxy ranking transfers unchanged.",
        "",
        "[Full model and training setup](trajectory-learning.md) · [Full-arm experiment](learning-grasps-and-generalization.md) · [Reproduction guide](reproducibility.md)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Download checksummed inputs when needed, rerun inference, and rebuild the table/bootstrap",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reproduce:
        required = [ARTIFACTS / "dataset.json", ARTIFACTS / "dataset.npz"] + [
            ARTIFACTS / "models-v3" / (name + ".pt")
            for name in ("mlp", "fm", "mf", "imf")
        ]
        if not all(path.exists() for path in required):
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/download_models.py")],
                cwd=ROOT,
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "deskbuddy.trajectory.evaluate",
                "--dataset",
                str(ARTIFACTS),
                "--models",
                str(ARTIFACTS / "models-v3"),
                "--samples",
                "500",
            ],
            cwd=ROOT,
            check=True,
        )
        source = json.loads(
            (ARTIFACTS / "models-v3/evaluation.json").read_text(encoding="utf-8")
        )
        pairs = bootstrap_comparison()
        output = args.output or ROOT / "runs/deskbuddy/reproduction"
    else:
        source = json.loads(
            (ROOT / "docs/research/results/trajectory-learning.json").read_text(
                encoding="utf-8"
            )
        )
        pairs = json.loads(
            (ROOT / "docs/research/results/planner-comparison.json").read_text(
                encoding="utf-8"
            )
        )["paired_family_bootstrap"]
        output = args.output or ROOT / "docs/research"
    output.mkdir(parents=True, exist_ok=True)
    (output / "planner-comparison.md").write_text(
        render(source, pairs), encoding="utf-8"
    )
    result = dict(
        metrics=source["metrics"],
        paired_family_bootstrap=pairs,
        inference_rerun=args.reproduce,
        sampling_seed=1729,
        training_seeds=[20261125],
    )
    if args.reproduce:
        result["dataset_sha256"] = hashlib.sha256(
            (ARTIFACTS / "dataset.npz").read_bytes()
        ).hexdigest()
    json_output = (
        (output / "planner-comparison.json")
        if args.reproduce or args.output
        else output / "results/planner-comparison.json"
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output / 'planner-comparison.md'}")


if __name__ == "__main__":
    main()
