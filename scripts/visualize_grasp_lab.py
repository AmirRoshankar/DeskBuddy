"""Render measured synthetic grasp results. No hardware or camera access."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deskbuddy.learned.grasp_lab import unit
from deskbuddy.trajectory.learning import Dense, Field, sample

root = Path("runs/flowreach/box-grasps-v1")
report = json.loads((root / "evaluation.json").read_text())["summary"]
adaptive = json.loads(
    Path("runs/flowreach/adaptive-box-grasps-v1/report.json").read_text()
)["metrics"]
plt.style.use("default")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
names = ["mlp", "fm", "imf", "grasp_mf", "grasp_mf_5"]
labels = ["MLP", "FM · 8", "iMF · 1", "GraspMF · 1", "GraspMF · 5"]
colors = ["#bd862b", "#5077b7", "#287c6c", "#8c64b5", "#d87548"]
for ax, key, title in [
    (axes[0, 0], "geometric_valid_one", "One candidate"),
    (axes[0, 1], "geometric_valid_any", "Up to four candidates (MLP: one)"),
]:
    values = [report[n][key] * 100 for n in names]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(ylim=(0, 100), ylabel="Toy geometric validity (%)", title=title)
    ax.tick_params(axis="x", rotation=10)
    ax.spines[["top", "right"]].set_visible(False)
c = torch.tensor(
    [[0.024 / 0.05, 0.036 / 0.05, 0.020 / 0.05, np.cos(0.6), np.sin(0.6)]],
    dtype=torch.float32,
)
torch.set_num_threads(1)
torch.manual_seed(37)
angles = {}
with torch.inference_mode():
    for name in ["mlp", "imf"]:
        cp = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)
        model = (
            Dense(5, 8, cp["width"])
            if name == "mlp"
            else Field(cp["width"], condition_dim=5, state_dim=8)
        )
        model.load_state_dict(cp["state_dict"])
        p = sample(model, c.repeat(512, 1), torch.randn(512, 8), name)
        w, x, y, z = unit(p[:, 3:7]).T
        yaw = torch.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
        angles[name] = np.rad2deg((yaw.numpy() - 0.3 + np.pi / 2) % np.pi - np.pi / 2)
ax = axes[1, 0]
ax.hist(
    angles["imf"],
    bins=np.linspace(-90, 90, 37),
    density=True,
    color=colors[2],
    alpha=0.75,
    label="iMF samples",
)
ax.axvline(angles["mlp"][0], color=colors[0], linewidth=3, label="MLP output")
for angle in [-90, 0, 90]:
    ax.axvline(angle, color="#666666", linestyle="--", linewidth=1)
ax.set(
    title="One unseen box: jaw orientation modes",
    xlabel="Closing-axis yaw relative to box (degrees)",
    ylabel="Density",
)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1, 1]
values = []
errors = []
for name in ["adaptive_mf", "adaptive_imf"]:
    for mode in ["fresh_noise", "reuse_previous"]:
        values.append(adaptive[name][mode]["next_action_change_mm"])
        errors.append(adaptive[name][mode]["heldout_window_rmse"])
ax.bar(
    ["AMF\nfresh", "AMF\nreuse", "iMF\nfresh", "iMF\nreuse"],
    values,
    color=["#cc8154", "#44887c"] * 2,
)
ax.set(ylabel="Near-future change (mm)", title="Fixed-observation consistency probe")
other = ax.twinx()
other.scatter(range(4), errors, color="#4c4660", marker="D", label="Normalized error")
other.set(ylabel="Normalized window RMSE", ylim=(0, 0.7))
other.legend(frameon=False, loc="upper center")
ax.spines["top"].set_visible(False)
other.spines["top"].set_visible(False)
fig.suptitle(
    "Synthetic grasp learning: first experiment", fontsize=17, fontweight="bold"
)
fig.supxlabel(
    "Procedural box labels, no contact simulation or measured robot pickup success. Numbers after model names are NFEs.",
    fontsize=9,
)
fig.savefig("docs/assets/box-grasp-learning.png", dpi=170)
print("Saved docs/assets/box-grasp-learning.png")
