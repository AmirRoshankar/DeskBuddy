"""Summarize matched 20k-step MLP/FM/iMF training runs without pooling seeds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/deskbuddy/trajectory-seeds")
    )
    args = parser.parse_args()
    seeds = [20261125, 20261126, 20261127]
    rows = []
    for seed in seeds:
        folder = args.root / str(seed)
        evaluation = json.loads(
            (folder / "evaluation.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (folder / "training_summary.json").read_text(encoding="utf-8")
        )
        metrics = {m["label"]: m for m in evaluation["metrics"]}
        rows.append(
            dict(
                seed=seed,
                dataset_sha256=training["dataset_sha256"],
                steps=training["steps"],
                peak_gpu_bytes=training["peak_gpu_bytes"],
                checkpoint_sha256={
                    m: hashlib.sha256((folder / (m + ".pt")).read_bytes()).hexdigest()
                    for m in ("mlp", "fm", "imf")
                },
                metrics={
                    key: metrics[key]
                    for key in (
                        "mlp-1nfe-1c",
                        "fm-8nfe-1c",
                        "imf-1nfe-1c",
                        "imf-1nfe-4c",
                    )
                },
            )
        )
    if len({r["dataset_sha256"] for r in rows}) != 1:
        raise ValueError("Seed runs must use identical data")
    summaries = {}
    for label in rows[0]["metrics"]:
        summaries[label] = {}
        for metric in ("raw_valid_rate", "blocked_raw_valid_rate"):
            values = np.array([r["metrics"][label][metric] for r in rows]) * 100
            summaries[label][metric] = dict(
                mean_percent=float(values.mean()),
                sample_sd_pp=float(values.std(ddof=1)),
            )
    output = Path("docs/research/results/training-seeds.json")
    output.write_text(
        json.dumps(
            dict(
                seeds=seeds,
                runs=rows,
                summary=summaries,
                sampling_seed=1729,
                test_conditions=500,
                layout_families=20,
                blocked_conditions=245,
                note="Three fresh MLP/FM/iMF runs; same data, recipe and evaluation noise. Original MF/iMF bootstrap is a separate fixed-checkpoint experiment.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Three-training-seed study",
        "",
        "Three fresh 20,000-step runs use the same synthetic dataset, splits, width, batch size, learning-rate schedule and objectives. Evaluation uses the same 500 conditions and sampling seed 1729.",
        "",
        "| Planner | Valid, mean ± sample SD | Blocked-route valid, mean ± sample SD |",
        "| --- | ---: | ---: |",
    ]
    for name, label in (
        ("MLP, one candidate", "mlp-1nfe-1c"),
        ("FM, 8 NFE", "fm-8nfe-1c"),
        ("iMF, 1 NFE", "imf-1nfe-1c"),
        ("iMF, four candidates", "imf-1nfe-4c"),
    ):
        a = summaries[label]["raw_valid_rate"]
        b = summaries[label]["blocked_raw_valid_rate"]
        lines.append(
            f"| {name} | {a['mean_percent']:.1f}% ± {a['sample_sd_pp']:.1f} pp | {b['mean_percent']:.1f}% ± {b['sample_sd_pp']:.1f} pp |"
        )
    lines += [
        "",
        "## Individual runs",
        "",
        "| Seed | MLP valid | iMF valid | iMF blocked-route valid | iMF, four candidates |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['seed']} | {m['mlp-1nfe-1c']['raw_valid_rate']:.1%} | {m['imf-1nfe-1c']['raw_valid_rate']:.1%} | {m['imf-1nfe-1c']['blocked_raw_valid_rate']:.1%} | {m['imf-1nfe-4c']['raw_valid_rate']:.1%} |"
        )
    lines += [
        "",
        "The spread is a sample standard deviation over three training runs, not a 95% confidence interval. The original five-planner table and iMF-minus-MF bootstrap remain separate: these new runs omit the retired MF training baseline. This supports repeatability on the proxy benchmark, not full-arm or hardware success.",
        "",
        "All runs stay below 54 MB peak allocated CUDA tensor memory. Dataset and checkpoint fingerprints are recorded in [the structured results](results/training-seeds.json). See [reproduction commands](reproducibility.md#training-seed-study).",
        "",
    ]
    Path("docs/research/training-seeds.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
