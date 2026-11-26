"""Build the concise comparison from recorded metrics, without rerunning models."""

import json
from pathlib import Path

import numpy as np


def bootstrap_comparison():
    folder = Path("runs/flowreach/trajectories-v1/models-v3")
    data = np.load(folder / "evaluation-paths.npz")
    metadata = json.loads((folder.parent / "dataset.json").read_text(encoding="utf-8"))
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
            labels.index(f"imf-1nfe-{candidates}c"),
            labels.index(f"mf-1nfe-{candidates}c"),
        )
        difference = data["valid"][a].astype(float) - data["valid"][b].astype(float)
        grouped = np.array([difference[families == family].mean() for family in unique])
        interval = np.percentile(grouped[draws].mean(axis=1) * 100, [2.5, 97.5])
        result[str(candidates)] = dict(
            imf_minus_mf_pp=float(difference.mean() * 100),
            family_bootstrap_95ci_pp=interval.tolist(),
            layout_families=len(unique),
            resamples=10000,
        )
    return result


def main():
    source = json.loads(
        Path("docs/research/results/trajectory-learning.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = {m["label"]: m for m in source["metrics"]}
    pairs = bootstrap_comparison()
    rows = [
        ("RRT-Connect", "rrt-0nfe-1c"),
        ("MLP", "mlp-1nfe-1c"),
        ("FM (8 evaluations)", "fm-8nfe-1c"),
        ("MF (1 evaluation)", "mf-1nfe-1c"),
        ("iMF (1 evaluation)", "imf-1nfe-1c"),
    ]
    lines = [
        "# Which planner is best so far?",
        "",
        "These are results for 500 held-out synthetic conditions with fixed trained checkpoints and a fixed noise seed. Of those, 245 had an obstacle blocking the direct start-to-goal segment. They do not measure physical robot success.",
        "",
        "**Raw valid** means the planner's own complete 32-point path passed workspace and continuous segment checks against inflated 2D obstacle geometry, before fallback. The checker tests the connecting segments, not just the waypoint dots. Start and goal are enforced explicitly. It does not test the SO-101 links, measured tip accuracy, acceleration limits or real depth uncertainty.",
        "",
        "For four candidates, raw-valid rate is the fraction of conditions with **at least one valid candidate**. Rejected candidates are not executed. RRT fallback is counted separately.",
        "",
        "## One candidate per condition",
        "",
        "| Planner | Own path valid / 500 | Valid when direct route blocked | RRT fallback | Median total ms | P95 total ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in rows:
        m = metrics[key]
        lines.append(
            f"| {name} | {round(m['raw_valid_rate'] * 500)}/500 ({m['raw_valid_rate']:.1%}) | {m['blocked_raw_valid_rate']:.1%} | {m['fallback_rate']:.1%} | {m['median_total_ms']:.2f} | {m['p95_total_ms']:.2f} |"
        )
    lines += [
        "",
        "Total time includes generation, path validation and any RRT fallback. Timings are warmed CPU measurements with one Torch thread, excluding camera acquisition, perception, IK and actuation. A millisecond here does not mean a millisecond robot movement. All policies reached 100% final proxy validity after RRT fallback; that does not make their learned planners equally successful.",
        "",
        "## Four candidates per condition",
        "",
        "| Planner | At least one valid / 500 | Valid when direct route blocked | RRT fallback | Median total ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("FM (8 evaluations each)", "fm-8nfe-4c"),
        ("MF (1 evaluation each)", "mf-1nfe-4c"),
        ("iMF (1 evaluation each)", "imf-1nfe-4c"),
    ):
        m = metrics[key]
        lines.append(
            f"| {name} | {round(m['raw_valid_rate'] * 500)}/500 ({m['raw_valid_rate']:.1%}) | {m['blocked_raw_valid_rate']:.1%} | {m['fallback_rate']:.1%} | {m['median_total_ms']:.2f} |"
        )
    lines += [
        "",
        "The deterministic MLP does not gain alternative paths by being called four times with identical inputs. Four FM candidates cost 32 sample evaluations; four one-step MF/iMF candidates cost four. Batched calls reduce overhead, but all candidates still need checking.",
        "",
        "## Practical ranking",
        "",
        "- **Most reliable on these simple scenes:** RRT, with 500/500 own paths valid. It is the classical reference and recovery mechanism, not a neural model.",
        "- **Strongest single-candidate neural result:** iMF, 431/500 versus MF's 409/500. Their total median times, 0.53 and 0.54 ms, are effectively too close to rank from this single timing pass.",
        "- **Four-candidate MF versus iMF:** close. iMF succeeds on only three additional cases, while MF is slightly faster in this run. Neither is a convincing overall winner at this budget.",
        "- **No universal method winner:** this is one training seed and a small family of synthetic tasks. iMF uses an auxiliary velocity network during training; inference generator capacity is matched, but training cost is higher.",
        "",
        f"A paired bootstrap over the 20 held-out obstacle-layout families estimates iMF minus MF at {pairs['1']['imf_minus_mf_pp']:.1f} percentage points for one candidate, with a 95% interval of {pairs['1']['family_bootstrap_95ci_pp'][0]:.1f} to {pairs['1']['family_bootstrap_95ci_pp'][1]:.1f}. With four candidates, the estimate is {pairs['4']['imf_minus_mf_pp']:.1f} points, interval {pairs['4']['family_bootstrap_95ci_pp'][0]:.1f} to {pairs['4']['family_bootstrap_95ci_pp'][1]:.1f}; this includes a tie or reversal. These intervals condition on the fixed checkpoints/noise seed and do not include training-seed variation.",
        "",
        "## Other metrics and what is missing",
        "",
        "The full report records generation and validation times separately, path length, minimum inflated-obstacle clearance, checkpoint selection score and NFE. Length/clearance summaries only include accepted model paths, so they cannot fairly crown a winner alone. Geometric second differences are recorded internally, but physical acceleration/jerk require time-parameterized joint trajectories. Goal endpoints are fixed by construction; measured physical positioning error and motor energy are not evaluated by this 2D experiment.",
        "",
        "## Live status",
        "",
        "All five planners can now receive the same live RGB-detected rectangular obstacles in a preview-only comparison. Object detection is shared, not a separate detector for each planner. YOLOE-26 Nano RGB instance masks define outlines; filtered depth estimates 3D boxes. The calibrated flat paper and modeled robot are excluded. The models have two obstacle slots; extra detections are conservatively merged into two rectangular regions, and every planner receives those same regions. Swapping objects updates detections and invalidates old previews.",
        "",
        "The first camera-fed test with four detections grouped into two regions accepted 0/1 MLP candidates and 0/4 FM, MF and iMF candidates; RRT found a path. This is one script-selected target in an estimated scene, not a hardware success-rate benchmark. It exposes domain shift and possible perception/grouping conservatism rather than proving that the offline ranking transfers to the desk.",
        "",
        "The main sidebar now selects RRT, MLP, FM, MF or iMF and one/four candidates for Go. Its route is lifted through numerical waypoint IK, time-parameterized, checked against joint/table/workspace and approximate arm-link/obstacle constraints, and executed without substituting a direct reach. Changed confirmed scene geometry stops execution. This executor has offline/fake-bus validation; physical Cartesian-route trials have not yet been run. Compare planners remains preview-only. Direct is a separate endpoint-only baseline blocked by detected objects.",
        "",
        "[Full offline metrics and plots](trajectory-learning.md) · [How the Go button works](controller-guide.md)",
        "",
    ]
    Path("docs/research/planner-comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    Path("docs/research/results/planner-comparison.json").write_text(
        json.dumps(
            dict(metrics=source["metrics"], paired_family_bootstrap=pairs), indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
