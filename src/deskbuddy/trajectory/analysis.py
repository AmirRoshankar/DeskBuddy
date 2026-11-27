"""Publish compact held-out trajectory metrics and unfiltered sample plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from .collision import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="runs/deskbuddy/trajectories-v1/models-v3")
    args = parser.parse_args()
    folder = Path(args.models)
    report = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    training = json.loads(
        (folder / "training_summary.json").read_text(encoding="utf-8")
    )
    dataset = folder.parent
    metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    data = np.load(folder / "evaluation-paths.npz")
    labels = data["labels"].tolist()
    selections = []
    families = set()
    for case_id in data["scenario_index"]:
        case = metadata["scenarios"][case_id]
        if case["direct_blocked"] and case["family"] not in families:
            selections.append(case)
            families.add(case["family"])
        if len(selections) == 2:
            break
    figure, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    figure.suptitle(
        "Held-out obstacle-conditioned trajectory predictions\nUnfiltered single samples · endpoints fixed · first two blocked layout families",
        fontsize=15,
    )
    for row, case in enumerate(selections):
        case_index = np.flatnonzero(data["scenario_index"] == case["index"])[0]
        scene = Scene(**case["scene"])
        for col, (label, title) in enumerate(
            (
                ("mlp-1nfe-1c", "MLP · 1 evaluation"),
                ("fm-8nfe-1c", "Flow matching · 8 evaluations"),
                ("mf-1nfe-1c", "MeanFlow · 1 evaluation"),
                ("imf-1nfe-1c", "iMF · 1 evaluation"),
            )
        ):
            axis = axes[row, col]
            variant = labels.index(label)
            path = data["raw"][variant, case_index]
            valid = scene.path_free(path)
            for cx, cy, radius in scene.circles:
                axis.add_patch(
                    Circle(
                        (cx * 100, cy * 100),
                        (radius + scene.inflation) * 100,
                        fill=False,
                        edgecolor="#777777",
                        linestyle=":",
                    )
                )
                axis.add_patch(
                    Circle((cx * 100, cy * 100), radius * 100, color="#424249")
                )
            for xmin, ymin, xmax, ymax in scene.boxes:
                d = scene.inflation
                axis.add_patch(
                    Rectangle(
                        ((xmin - d) * 100, (ymin - d) * 100),
                        (xmax - xmin + 2 * d) * 100,
                        (ymax - ymin + 2 * d) * 100,
                        fill=False,
                        edgecolor="#777777",
                        linestyle=":",
                    )
                )
                axis.add_patch(
                    Rectangle(
                        (xmin * 100, ymin * 100),
                        (xmax - xmin) * 100,
                        (ymax - ymin) * 100,
                        color="#424249",
                    )
                )
            axis.plot(
                path[:, 0] * 100,
                path[:, 1] * 100,
                color="#277AA4" if valid else "#C2472D",
                marker=".",
                markersize=3,
                linewidth=1.5,
            )
            axis.scatter(*np.array(case["start"]) * 100, c="#246B49", s=35, zorder=4)
            axis.scatter(
                *np.array(case["goal"]) * 100, c="#AD3D35", marker="*", s=50, zorder=4
            )
            axis.set(
                xlim=(0, 21),
                ylim=(0, 29.7),
                xlabel="X (cm)",
                ylabel="Y (cm)",
                title=f"{title}\nCase {case['index']} · {'valid' if valid else 'rejected'}",
            )
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
    figure.savefig("docs/assets/trajectory-learning.png", dpi=150)
    plt.close(figure)
    report["training"] = training
    report["figure_selection"] = (
        "First two held-out blocked layout families, without selecting for model success"
    )
    Path("docs/research/results/trajectory-learning.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Obstacle-conditioned trajectory learning",
        "",
        "This is a separate offline trajectory experiment, not the live Flow + IK endpoint model. Four models predict all 30 interior XY waypoints; start and goal are fixed exactly. The input contains start, goal, up to two obstacles and tool/margin inflation. Normalized circle radii retain both axis scales; all validation happens after returning to physical metres.",
        "",
        "## Held-out comparison",
        "",
        "| Method | NFE / candidate | Candidates | Raw valid | Blocked-case valid | RRT fallback | Median total ms | P95 total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in report["metrics"]:
        lines.append(
            f"| {m['method'].upper()} | {m['nfe_per_candidate']} | {m['candidates']} | {m['raw_valid_rate']:.1%} | {m['blocked_raw_valid_rate']:.1%} | {m['fallback_rate']:.1%} | {m['median_total_ms']:.2f} | {m['p95_total_ms']:.2f} |"
        )
    lines += [
        "",
        f"Each row uses the same {len(report['test_scenario_ids'])} held-out conditions. Total time includes neural preprocessing/generation/decoding, continuous segment validation, and RRT fallback when all candidates are invalid. It excludes perception and robot execution. Raw validity is before fallback. K=4 batches four candidate trajectories, then selects the shortest valid one; this costs K times the candidate NFEs and additional checking. It is not a free one-sample improvement.",
        "",
        "![Unfiltered samples on held-out obstacles](trajectory-learning.png)",
        "",
        "## Training and limitations",
        "",
        "The experiment uses 8,000 training and 1,000 validation paths; all layout families and route seeds stay in their assigned split. Networks have three 192-unit hidden layers. The conditional MLP has 89,340 parameters; each flow generator has 101,244 inference parameters. iMF adds a separate supervised instantaneous-velocity network during training, bringing its training parameter count to 202,296; that network is not called during sampling.",
        "",
        f"The final run used {training['steps']:,} steps on an RTX 5070, with peak allocated tensor memory {training['peak_gpu_bytes'] / 1e6:.1f} MB. MF/iMF use detached adaptive weighting (p=0.5), 25% diagonal examples after a 1,000-step diagonal warm-up, and the same cosine learning-rate schedule starting at 0.0003. Checkpoints were selected using validation distribution energy score, a statistical score unrelated to motor electrical energy. These are engineering adaptations, not reproductions of the papers' image-generation recipes.",
        "",
        "The initial plain-L2 run had severe iMF divergence; the second adaptive-p=1 run was stable but weak. Both histories and checkpoints are preserved under models-v1 and models-v2. The reported models-v3 run uses p=0.5 and the iMF auxiliary velocity head. Comparisons between these runs confound multiple recipe changes and are not controlled ablations. No universal iMF advantage is claimed.",
        "",
        "The MeanFlow objective follows [Geng et al., Algorithm 1](https://arxiv.org/html/2505.13447v1). iMF uses the predicted-velocity JVP formulation and optional auxiliary head described in [Geng et al., Algorithm 1 and Appendix A](https://arxiv.org/html/2512.02012v2). Standard FM uses instantaneous-velocity regression with Euler sampling; MF/iMF use interval-average updates. MF and iMF have genuinely different training tangents.",
        "",
        "No paths from this experiment are sent to motors. This 2D tool proxy has no full-arm collision checks, IK lifting, acceleration limits or perception uncertainty model. Physical motor energy cannot be inferred from these un-timed XY paths. The live UI still uses the earlier endpoint-IK model. Next: validate obstacle perception, lift paths continuously into joint space, then supervised static-obstacle trials before dynamic stop-and-replan.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python -m deskbuddy.trajectory.train --output runs/deskbuddy/trajectories-v1/models-v3 --steps 20000 --lr 0.0003 --diagonal-fraction 0.25 --adaptive-power 0.5 --imf-auxiliary-head",
        "python -m deskbuddy.trajectory.evaluate --models runs/deskbuddy/trajectories-v1/models-v3",
        "python -m deskbuddy.trajectory.analysis --models runs/deskbuddy/trajectories-v1/models-v3",
        "```",
        "",
        "Training refuses to overwrite checkpoints: use a new output directory for a new run. [Structured metrics](trajectory-learning.json) include all timing components and checkpoint steps.",
        "",
    ]
    Path("docs/research/trajectory-learning.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Exported docs/research/trajectory-learning.md, .json and .png")


if __name__ == "__main__":
    main()
