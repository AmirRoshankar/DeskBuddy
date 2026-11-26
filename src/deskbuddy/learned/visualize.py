"""Export scientific figures and a standalone interactive learned-field report."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..kinematics import Arm
from ..planning import check_workspace, plan_reach
from .data import check_budget
from .inference import Proposal

COLORS = ["#ff9275", "#63d7c4", "#b9a4ff", "#f0cb68"]
NAMES = ["Shoulder pan", "Shoulder lift", "Elbow flex", "Wrist flex"]


def field_trace(wrapper, x, noise, steps=65):
    n = wrapper.normalization
    c = torch.tensor((x - n["x_mean"]) / n["x_std"], dtype=torch.float32)[None]
    progress = np.linspace(0, 1, steps)
    states = []
    velocities = []
    with torch.inference_mode():
        for s in progress:
            t = torch.full((1, 1), 1 - float(s))
            # Direct finite-interval map from the same noise at t=1 to r=1-s.
            z = noise - float(s) * wrapper.model(noise, t, torch.ones_like(t), c)
            velocity = -wrapper.model(z, t, t, c)
            q = x[:5].copy()
            q[:4] += z.numpy()[0] * n["y_std"] + n["y_mean"]
            states.append(q)
            velocities.append(np.rad2deg(velocity.numpy()[0] * n["y_std"]))
    return progress, np.array(states), np.array(velocities)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="runs/flowreach")
    parser.add_argument(
        "--output", default="docs/research/flowreach-visualizations.html"
    )
    args = parser.parse_args()
    folder = Path(args.folder)
    check_budget(folder, 20_000_000)
    meta = json.loads((folder / "dataset.json").read_text())
    benchmark = json.loads((folder / "benchmark.json").read_text())
    config = meta["config"]
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    wrapper = Proposal(folder / "imf.pt", config)
    inputs = np.load(folder / "dataset.npz")["test_x"].astype(float)
    index = next(
        i
        for i, r in enumerate(benchmark["rows"]["imf_refined"])
        if r.get("path_valid")
        and any(m["used"] == "flow_assisted" for m in r.get("planner_info", []))
    )
    case_index = index
    x = inputs[index]
    progress, states, velocities = field_trace(wrapper, x, wrapper.noise)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"type": "scene"}, {}]],
        subplot_titles=(
            "Joint angles during virtual denoising",
            "Learned denoising velocity — NOT motor speed",
            "Tip projection: virtual generation vs physical reference",
            "Raw iMF error on an XYZ slice; fixed starting pose",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.13,
    )
    for j, name in enumerate(NAMES):
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=np.rad2deg(states[:, j]),
                name=name,
                line_color=COLORS[j],
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=progress,
                y=velocities[:, j],
                name=name,
                line_color=COLORS[j],
                legendgroup=name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    virtual_paths = []
    for seed in range(8):
        noise = torch.randn((1, 4), generator=torch.Generator().manual_seed(seed))
        _, poses, _ = field_trace(wrapper, x, noise)
        tips = np.array([arm.fk(q) for q in poses])
        virtual_paths.append(tips)
        fig.add_trace(
            go.Scatter3d(
                x=tips[:, 0],
                y=tips[:, 1],
                z=tips[:, 2],
                mode="lines",
                name="Virtual generative paths",
                showlegend=seed == 0,
                line=dict(color="#b9a4ff", width=2),
                opacity=0.45,
            ),
            row=2,
            col=1,
        )
    reference = plan_reach(
        arm, x[:5], x[5:], dict(config, planner_mode="numerical", control_hz=20)
    )
    physical = np.array([arm.fk(q) for q in reference.joints_rad])
    fig.add_trace(
        go.Scatter3d(
            x=physical[:, 0],
            y=physical[:, 1],
            z=physical[:, 2],
            mode="lines",
            name="Physical planned reference",
            line=dict(color="#63d7c4", width=6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x[5]],
            y=[x[6]],
            z=[x[7]],
            mode="markers",
            name="Target",
            marker=dict(size=6, color="#ff9275", symbol="diamond"),
        ),
        row=2,
        col=1,
    )
    xs = np.linspace(config["workspace_min_m"][0], config["workspace_max_m"][0], 27)
    ys = np.linspace(config["workspace_min_m"][1], config["workspace_max_m"][1], 27)
    heights = [0.06, 0.10, 0.14]
    errors = []
    heat_indices = []
    for h in heights:
        grid = np.full((len(ys), len(xs)), np.nan)
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                target = np.array([cx, cy, h])
                try:
                    check_workspace(target, config)
                    q = wrapper.predict(x[:5], target)
                    if ((q < arm.lower) | (q > arm.upper)).any():
                        continue
                    grid[iy, ix] = np.linalg.norm(arm.fk(q) - target) * 1000
                except ValueError:
                    pass
        errors.append(grid)
        heat_indices.append(len(fig.data))
        fig.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=grid,
                zmin=0,
                zmax=15,
                colorscale="Viridis",
                visible=h == heights[1],
                colorbar=dict(title="FK gap<br>mm", len=0.37, y=0.2),
                name=f"z={h:.2f} m",
                hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>gap=%{z:.2f} mm<extra></extra>",
            ),
            row=2,
            col=2,
        )
    buttons = []
    for i, h in enumerate(heights):
        visible = [True] * len(fig.data)
        for j, trace_index in enumerate(heat_indices):
            visible[trace_index] = j == i
        buttons.append(
            dict(
                label=f"Heatmap z = {h * 100:.0f} cm",
                method="update",
                args=[dict(visible=visible)],
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#171719",
        plot_bgcolor="#202023",
        height=950,
        title="FlowReach · learned generative geometry",
        margin=dict(t=110, b=65),
        legend=dict(orientation="h", y=1.12),
        updatemenus=[dict(buttons=buttons, active=1, x=1, y=0.47, xanchor="right")],
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )
    fig.update_xaxes(title_text="Denoising progress (dimensionless)", row=1, col=1)
    fig.update_yaxes(title_text="Joint angle (deg)", row=1, col=1)
    fig.update_xaxes(title_text="Denoising progress (dimensionless)", row=1, col=2)
    fig.update_yaxes(title_text="deg / unit denoising progress", row=1, col=2)
    fig.update_xaxes(title_text="Target X (m)", row=2, col=2)
    fig.update_yaxes(title_text="Target Y (m)", row=2, col=2)

    # Two-dimensional section through a four-dimensional generative vector field.
    n = wrapper.normalization
    c = torch.tensor((x - n["x_mean"]) / n["x_std"], dtype=torch.float32)[None]
    center = (states[len(states) // 2, :4] - x[:4] - n["y_mean"]) / n["y_std"]
    values = np.linspace(-1.5, 1.5, 17)
    xx, yy = np.meshgrid(values, values)
    latent = np.tile(center, (xx.size, 1))
    latent[:, :2] = np.c_[xx.ravel(), yy.ravel()]
    with torch.inference_mode():
        z = torch.tensor(latent, dtype=torch.float32)
        t = torch.full((len(z), 1), 0.5)
        field = -wrapper.model(z, t, t, c.repeat(len(z), 1)).numpy()
    coords = np.rad2deg(x[:4] + latent * n["y_std"] + n["y_mean"])
    vectors = np.rad2deg(field * np.array(n["y_std"])) * 0.06
    lines_x = []
    lines_y = []
    for point, vector in zip(coords[:, :2], vectors[:, :2]):
        end = point + vector
        for a, b in (
            (point, end),
            (end, end - 0.2 * vector + 0.1 * np.array([-vector[1], vector[0]])),
            (end, end - 0.2 * vector - 0.1 * np.array([-vector[1], vector[0]])),
        ):
            lines_x.extend([a[0], b[0], None])
            lines_y.extend([a[1], b[1], None])
    field_fig = go.Figure(
        go.Scatter(
            x=lines_x,
            y=lines_y,
            mode="lines",
            line=dict(color="#ff9275", width=1),
            showlegend=False,
        )
    )
    field_fig.update_layout(
        template="plotly_dark",
        height=580,
        title="Generative velocity-field slice at flow time t = 0.5 (other two latent coordinates fixed)",
        xaxis_title="Virtual shoulder pan (deg)",
        yaxis_title="Virtual shoulder lift (deg)",
        paper_bgcolor="#171719",
        plot_bgcolor="#202023",
    )

    energy = json.loads((folder / "energy_comparison.json").read_text())
    energy_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Matched-case median motor energy (estimated)",
            "Example total motor input power (estimated)",
        ),
    )
    method_names = ["Numerical", "MLP + IK", "iMF + IK"]
    energy_fig.add_trace(
        go.Bar(
            x=method_names,
            y=[
                energy["summary"][m]["median_energy_j"]
                for m in ("numerical", "mlp", "imf")
            ],
            marker_color=["#f5f3f0", "#63d7c4", "#ff9275"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for name, label, color in zip(
        ("numerical", "mlp", "imf"), method_names, ("#f5f3f0", "#63d7c4", "#ff9275")
    ):
        e = energy["example_power_curves"][name]
        energy_fig.add_trace(
            go.Scatter(
                x=e["time_s"], y=e["total_power_w"], name=label, line_color=color
            ),
            row=1,
            col=2,
        )
    energy_fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#171719",
        plot_bgcolor="#202023",
        legend=dict(orientation="h", y=1.18),
    )
    energy_fig.update_yaxes(title_text="Motor energy per reach (J)", row=1, col=1)
    energy_fig.update_yaxes(title_text="Total input power (W)", row=1, col=2)
    energy_fig.update_xaxes(title_text="Planned physical time (s)", row=1, col=2)

    multimodal = json.loads((folder / "multimodal/evaluation.json").read_text())
    e = multimodal["example"]
    mode_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "One target, two valid teacher postures",
            "Raw predicted wrist-flex distribution for that target",
        ),
    )
    for name, color in (("imf", "#ff9275"), ("mlp", "#63d7c4"), ("teacher", "#f5f3f0")):
        poses = np.array(e[name + "_poses"])
        bounded = ((poses >= arm.lower) & (poses <= arm.upper)).all(1)
        mode_fig.add_trace(
            go.Scatter(
                x=np.rad2deg(poses[:, 2]),
                y=np.rad2deg(poses[:, 3]),
                mode="markers",
                name=name,
                marker=dict(
                    color=[color if valid else "#929292" for valid in bounded],
                    size=12 if name != "imf" else 5,
                    symbol=[
                        ("diamond" if name == "teacher" else "circle") if valid else "x"
                        for valid in bounded
                    ],
                ),
            ),
            row=1,
            col=1,
        )
        if name == "imf":
            mode_fig.add_trace(
                go.Histogram(
                    x=np.rad2deg(poses[:, 3]),
                    name="iMF samples",
                    marker_color=color,
                    nbinsx=25,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        if name == "mlp":
            mode_fig.add_vline(
                x=float(np.rad2deg(poses[0, 3])), line_color=color, row=1, col=2
            )
    mode_fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="#171719",
        plot_bgcolor="#202023",
    )
    mode_fig.update_xaxes(title_text="Elbow flex (deg)", row=1, col=1)
    mode_fig.update_yaxes(title_text="Wrist flex (deg)", row=1, col=1)
    mode_fig.update_xaxes(title_text="Wrist flex (deg)", row=1, col=2)
    mode_fig.update_yaxes(title_text="Sample count", row=1, col=2)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        (
            "Learned geometry",
            "Fixed start and target. Purple curves are unconstrained virtual denoising paths, never motor commands. Teal is the validated physical reference. Heatmap colors clip at 15 mm; hover shows actual errors. Blanks are outside selection/training coverage or violate joint bounds. This is not a full reachability map.",
            fig,
        ),
        (
            "Velocity field",
            "A 2D section of the four-dimensional learned field. Arrows show decreasing-flow-time direction, scaled by 0.06. They are not physical joint velocities.",
            field_fig,
        ),
        (
            "Motor energy",
            f"Quasi-static estimate at {energy['voltage_v']:.1f} V; {energy['matched_valid_cases']} matched cases. Five powered joints, empty gripper motor off. Similar curves are expected because refined trajectories are nearly identical. Model uncertainty is much larger than the differences between methods.",
            energy_fig,
        ),
        (
            "Controlled multimodality test",
            "Separate exploratory model; not the deployed checkpoint. Same XYZ has balanced 40°/90° wrist-flex teacher solutions. iMF can sample both modes; a deterministic squared-error regressor tends toward their mean. Gray crosses violate joint limits. All raw samples still require path checks and refinement. A mixture-density MLP could also represent multiple modes.",
            mode_fig,
        ),
    ]
    html = "<!doctype html><html><head><meta charset='utf-8'><title>FlowReach learned-field report</title></head><body style='background:#171719;color:#f5f3f0;font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 24px'><h1>FlowReach · learned fields and motor-energy estimates</h1><p>Offline model analysis. No hardware access. Generative time and physical motion time are different.</p>"
    for index, (title, caption, plot) in enumerate(sections):
        html += (
            f"<h2>{title}</h2><p style='color:#b1afad;max-width:1100px'>{caption}</p>"
            + pio.to_html(
                plot,
                include_plotlyjs=index == 0,
                full_html=False,
                config=dict(responsive=True, displaylogo=False),
            )
        )
    html += "</body></html>"
    check_budget(folder, len(html.encode()))
    output.write_text(html, encoding="utf-8")

    # Standard static scientific export for visual QA and sharing.
    plt.style.use("dark_background")
    overview, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    overview.set_facecolor("#171719")
    for axis in axes.ravel():
        axis.set_facecolor("#202023")
    for j, name in enumerate(NAMES):
        axes[0, 0].plot(progress, np.rad2deg(states[:, j]), label=name, color=COLORS[j])
        axes[0, 1].plot(progress, velocities[:, j], label=name, color=COLORS[j])
    axes[0, 0].set(
        title="Virtual joint samples during denoising",
        xlabel="Denoising progress (not physical time)",
        ylabel="Joint angle (deg)",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        title="Learned denoising velocities — not motor speeds",
        xlabel="Denoising progress",
        ylabel="deg / unit denoising progress",
    )
    heat = axes[1, 0].pcolormesh(
        xs, ys, errors[1], cmap="viridis", vmin=0, vmax=15, shading="auto"
    )
    overview.colorbar(heat, ax=axes[1, 0], label="Raw FK error (mm)")
    axes[1, 0].set(
        title="Raw endpoint error, fixed start, target Z = 10 cm",
        xlabel="Target X (m)",
        ylabel="Target Y (m)",
    )
    poses = np.array(e["imf_poses"])
    bounded = ((poses >= arm.lower) & (poses <= arm.upper)).all(1)
    teacher = np.array(e["teacher_poses"])
    mlp = np.array(e["mlp_poses"])
    axes[1, 1].scatter(
        np.rad2deg(poses[bounded, 2]),
        np.rad2deg(poses[bounded, 3]),
        s=12,
        color="#ff9275",
        label="iMF samples",
    )
    axes[1, 1].scatter(
        np.rad2deg(poses[~bounded, 2]),
        np.rad2deg(poses[~bounded, 3]),
        s=24,
        color="#929292",
        marker="x",
        label="Joint-limit violation",
    )
    axes[1, 1].scatter(
        np.rad2deg(teacher[:, 2]),
        np.rad2deg(teacher[:, 3]),
        s=70,
        color="white",
        marker="D",
        label="Two valid teacher postures",
    )
    axes[1, 1].scatter(
        np.rad2deg(mlp[:, 2]),
        np.rad2deg(mlp[:, 3]),
        s=70,
        color="#63d7c4",
        marker="x",
        label="Deterministic regressor",
    )
    axes[1, 1].set(
        title="Separate two-posture stress test, same XYZ",
        xlabel="Elbow flex (deg)",
        ylabel="Wrist flex (deg)",
    )
    axes[1, 1].legend(fontsize=8)
    png = output.with_suffix(".png")
    overview.savefig(png, dpi=150)
    plt.close(overview)
    print(
        json.dumps(
            dict(
                html=str(output),
                png=str(png),
                selected_case=case_index,
                heatmap_valid_counts=[int(np.isfinite(g).sum()) for g in errors],
                plots=len(sections),
            )
        )
    )


if __name__ == "__main__":
    main()
