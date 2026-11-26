"""Non-modal playground editor; applying never moves the robot."""

import tkinter as tk

import numpy as np

from .theme import COLORS, FONT_DATA, FONT_SMALL
from .widgets import ControlButton


def edited_workspace(config, lower_cm, upper_cm, restrict, show):
    bounds = np.asarray([lower_cm, upper_cm], float) / 100
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Enter six finite numbers in cm")
    if (bounds[1] - bounds[0] < 0.001).any():
        raise ValueError("Each maximum must exceed its minimum by at least 0.1 cm")
    return dict(
        config,
        workspace_min_m=bounds[0].tolist(),
        workspace_max_m=bounds[1].tolist(),
        restrict_to_test_bed=bool(restrict),
        show_workspace=bool(show),
    )


def open_editor(root, config, on_apply):
    window = tk.Toplevel(root)
    window.title("Playground bounds")
    window.configure(bg=COLORS.canvas)
    window.resizable(False, False)
    window.transient(root)
    body = tk.Frame(window, bg=COLORS.canvas, padx=16, pady=16)
    body.pack(fill="both", expand=True)
    for col, title in enumerate(("Robot base · cm", "Minimum", "Maximum")):
        tk.Label(
            body, text=title, bg=COLORS.canvas, fg=COLORS.muted, font=FONT_SMALL
        ).grid(row=0, column=col, padx=8, pady=(0, 10))
    values = []
    for axis, label in enumerate(("X · forward", "Y · sideways", "Z · up")):
        tk.Label(body, text=label, bg=COLORS.canvas, fg=COLORS.text).grid(
            row=axis + 1, column=0, sticky="w", padx=8
        )
        row = []
        for col, key in enumerate(("workspace_min_m", "workspace_max_m"), 1):
            var = tk.StringVar(value=f"{config[key][axis] * 100:.3f}")
            tk.Entry(
                body,
                textvariable=var,
                width=10,
                bg=COLORS.raised,
                fg=COLORS.text,
                insertbackground=COLORS.text,
                highlightbackground=COLORS.border,
                highlightcolor=COLORS.accent_light,
                highlightthickness=1,
                font=FONT_DATA,
                relief="flat",
            ).grid(row=axis + 1, column=col, padx=8, pady=6, ipady=6)
            row.append(var)
        values.append(row)
    restrict = tk.BooleanVar(value=config.get("restrict_to_test_bed", True))
    show = tk.BooleanVar(value=config.get("show_workspace", True))
    for row, text, var in (
        (4, "Also restrict targets to A4 inset", restrict),
        (5, "Show 3D bounds in camera", show),
    ):
        tk.Checkbutton(
            body,
            text=text,
            variable=var,
            bg=COLORS.canvas,
            fg=COLORS.text,
            selectcolor=COLORS.raised,
            activebackground=COLORS.canvas,
            activeforeground=COLORS.text,
            highlightbackground=COLORS.canvas,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=5)
    tk.Label(
        body,
        text="Joint travel and table clearance still apply.\nApplying clears the preview; it does not move the arm.",
        bg=COLORS.canvas,
        fg=COLORS.muted,
        font=FONT_SMALL,
        justify="left",
    ).grid(row=6, column=0, columnspan=3, sticky="w", pady=10)
    error = tk.StringVar()
    tk.Label(
        body,
        textvariable=error,
        bg=COLORS.canvas,
        fg=COLORS.warning,
        wraplength=340,
        justify="left",
    ).grid(row=7, column=0, columnspan=3, sticky="w")

    def apply():
        try:
            updated = edited_workspace(
                config,
                [float(v[0].get()) for v in values],
                [float(v[1].get()) for v in values],
                restrict.get(),
                show.get(),
            )
            on_apply(updated)
        except (ValueError, OSError, RuntimeError) as exc:
            error.set(str(exc))
            return
        window.destroy()

    ControlButton(body, text="Apply bounds", command=apply, primary=True).grid(
        row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0)
    )
    return window
