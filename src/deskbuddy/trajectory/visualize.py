"""Export the offline teacher-path benchmark; never accesses camera or motors."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/flowreach/trajectories-v1")
    parser.add_argument("--output", default="docs/assets/trajectory-baseline.png")
    args = parser.parse_args()
    folder = Path(args.folder)
    report = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    data = np.load(folder / "dataset.npz")
    paths, indices = data["paths"], data["scenario_index"]
    candidates = []
    for scene in report["scenarios"]:
        where = np.flatnonzero(indices == scene["index"])
        if scene["direct_blocked"] and len(where) >= 2:
            difference = float(
                np.linalg.norm(paths[where[0]] - paths[where[1]], axis=1).max()
            )
            candidates.append((difference, scene, where[:2]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected, families = [], set()
    for item in candidates:
        if item[1]["family"] not in families:
            selected.append(item)
            families.add(item[1]["family"])
        if len(selected) == 6:
            break
    figure, axes = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
    figure.suptitle(
        "Offline RRT-Connect teacher trajectories\nSame conditions, two random seeds · 32 waypoints · 2D tool proxy",
        fontsize=16,
    )
    for axis, item in zip(axes.flat, selected):
        _, case, where = item
        scene = case["scene"]
        inflation = scene["tool_radius"] + scene["margin"]
        for cx, cy, radius in scene["circles"]:
            axis.add_patch(
                Circle(
                    (cx * 100, cy * 100),
                    (radius + inflation) * 100,
                    fill=False,
                    edgecolor="#777777",
                    linestyle=":",
                )
            )
            axis.add_patch(Circle((cx * 100, cy * 100), radius * 100, color="#424249"))
        for xmin, ymin, xmax, ymax in scene["boxes"]:
            axis.add_patch(
                Rectangle(
                    ((xmin - inflation) * 100, (ymin - inflation) * 100),
                    (xmax - xmin + 2 * inflation) * 100,
                    (ymax - ymin + 2 * inflation) * 100,
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
        for index, color, label in zip(
            where, ("#C2472D", "#277AA4"), ("Seed A", "Seed B")
        ):
            path = paths[index] * 100
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                marker=".",
                markersize=3,
                linewidth=1.6,
                label=label,
            )
        start, goal = np.array(case["start"]) * 100, np.array(case["goal"]) * 100
        axis.scatter(*start, c="#246B49", s=45, marker="o", zorder=4)
        axis.scatter(*goal, c="#AD3D35", s=55, marker="*", zorder=4)
        axis.set(
            xlim=(0, 21),
            ylim=(0, 29.7),
            xlabel="X (cm)",
            ylabel="Y (cm)",
            title=f"Case {case['index']} · {case['split']}",
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    summary = dict(
        metrics=report["metrics"],
        split_counts=dict(Counter(data["split"].tolist())),
        attempt_status_counts=dict(
            Counter(row["status"] for row in report["attempts"])
        ),
        split_method=report["split_method"],
        limitations=report["limitations"],
        figure_selection="Six cases with largest seed-to-seed deviation, with distinct layout families; illustrative extremes, not representative average quality",
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
