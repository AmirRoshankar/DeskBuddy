"""Resizable native dashboard composition; no hardware operations."""

import tkinter as tk
from dataclasses import dataclass

from .hover import HoverControl
from .theme import COLORS, FONT_SMALL, apply
from .widgets import ControlButton, EventLog, panel


@dataclass
class Dashboard:
    color: tk.Label
    depth: tk.Label
    diagram: tk.Label
    zoom: tk.Label
    depth_zoom: tk.Label
    status: tk.StringVar
    fault_status: tk.StringVar
    cursor_info: tk.StringVar
    torque_status: tk.StringVar
    telemetry_status: tk.StringVar
    go: tk.Button | None
    help_label: tk.Label
    fault_label: tk.Label
    display_w: int
    display_h: int
    diagram_h: int
    hover: HoverControl
    disable: tk.Button | None
    webcam: tk.Label | None = None
    path_status: dict | None = None


def build(
    root,
    *,
    initial_status,
    controls,
    calibration_mode,
    on_go,
    on_stop,
    on_recalibrate,
    hover_m=0.12,
    on_hover=lambda _: None,
    planner_mode="numerical",
    on_planner=lambda _: None,
    on_disable=lambda: None,
    on_bounds=lambda: None,
    on_compare=lambda: None,
    trajectory_method="rrt",
    candidates=1,
    on_trajectory=lambda method, count: None,
    on_scene=lambda: None,
    on_collision_override=lambda value: None,
    on_grab_mode=lambda value: None,
    on_grab_keys=lambda: None,
    on_imf_grasp=lambda: None,
    on_home=lambda: None,
    on_calibrate_views=lambda: None,
    on_discard_zone=lambda: None,
    task_mode="reach",
):
    apply(root)
    root.title("DeskBuddy")
    root.minsize(1000, 680)
    root.geometry(
        f"{min(1400, root.winfo_screenwidth() - 64)}x{min(920, root.winfo_screenheight() - 96)}"
    )
    root.grid_columnconfigure(0, minsize=168)
    root.grid_columnconfigure(1, weight=3, minsize=420)
    root.grid_columnconfigure(2, weight=1, minsize=310)
    root.grid_rowconfigure(2, weight=4, minsize=360)
    root.grid_rowconfigure(3, weight=1, minsize=140)
    header = tk.Frame(root)
    header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))
    go = None
    disable = None
    if controls:
        actions = tk.Frame(header)
        actions.pack(side="left")
        go = ControlButton(
            actions,
            text="Go",
            command=on_go,
            primary=True,
            tone="go",
            state="disabled",
            width=5,
        )
        go.pack(side="left", padx=(0, 8))
        ControlButton(actions, text="Stop", command=on_stop, tone="stop", width=5).pack(
            side="left", padx=(0, 8)
        )
        disable = ControlButton(actions, text="Disable", command=on_disable, width=7)
        disable.pack(side="left", padx=(0, 8))
        ControlButton(actions, text="Recalibrate", command=on_recalibrate).pack(
            side="left"
        )
    ControlButton(header, text="Bounds", command=on_bounds).pack(side="right")
    ControlButton(header, text="Pop out 3D", command=on_scene).pack(
        side="right", padx=8
    )
    ControlButton(header, text="Calibrate views", command=on_calibrate_views).pack(
        side="right"
    )
    ControlButton(header, text="Compare planners", command=on_compare).pack(
        side="left", padx=(12, 0)
    )
    torque_status = tk.StringVar(value="Waiting for feedback")
    telemetry_status = tk.StringVar()
    meta = tk.Frame(root)
    meta.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
    tk.Label(
        meta, textvariable=torque_status, fg=COLORS.muted, font=FONT_SMALL, anchor="w"
    ).pack(side="left")
    tk.Label(
        meta,
        textvariable=telemetry_status,
        fg=COLORS.muted,
        font=FONT_SMALL,
        anchor="e",
    ).pack(side="right")
    sidebar = tk.Frame(root, width=152)
    sidebar.grid(row=2, column=0, rowspan=2, sticky="ns", padx=8, pady=(0, 8))
    sidebar.pack_propagate(False)
    tk.Label(sidebar, text="Planner", font=FONT_SMALL, fg=COLORS.muted).pack(anchor="w")
    methods = {
        "Direct": "direct",
        "RRT": "rrt",
        "MLP": "mlp",
        "FM": "fm",
        "iMF": "imf",
    }
    choice = tk.StringVar(
        value=next(k for k, v in methods.items() if v == trajectory_method)
    )
    count = tk.IntVar(value=candidates)
    if trajectory_method in ("direct", "mlp"):
        count.set(1)

    def select_route(*_):
        deterministic = methods[choice.get()] in ("direct", "mlp")
        if deterministic:
            count.set(1)
        for button in candidate_row.winfo_children():
            button.configure(state="disabled" if deterministic else "normal")
        on_trajectory(methods[choice.get()], count.get())

    menu = tk.OptionMenu(sidebar, choice, *methods, command=select_route)
    menu.configure(
        width=11,
        bg=COLORS.raised,
        fg=COLORS.text,
        highlightthickness=0,
        activebackground=COLORS.surface,
        activeforeground=COLORS.text,
        font=FONT_SMALL,
        bd=0,
    )
    menu["menu"].configure(bg=COLORS.raised, fg=COLORS.text, font=FONT_SMALL)
    menu.pack(fill="x", pady=(4, 8))
    tk.Label(sidebar, text="Candidates", font=FONT_SMALL, fg=COLORS.muted).pack(
        anchor="w"
    )
    candidate_row = tk.Frame(sidebar)
    candidate_row.pack(fill="x", pady=(2, 10))
    for n in (1, 4):
        tk.Radiobutton(
            candidate_row,
            text=str(n),
            variable=count,
            value=n,
            command=select_route,
            font=FONT_SMALL,
            fg=COLORS.text,
            bg=COLORS.canvas,
            selectcolor=COLORS.raised,
            activebackground=COLORS.raised,
            activeforeground=COLORS.text,
            highlightthickness=0,
            state="disabled" if trajectory_method in ("direct", "mlp") else "normal",
        ).pack(side="left", padx=(0, 12))
    collision_override = tk.BooleanVar(value=False)
    tk.Checkbutton(
        sidebar,
        text="Allow predicted\ncollisions",
        variable=collision_override,
        command=lambda: on_collision_override(collision_override.get()),
        font=FONT_SMALL,
        selectcolor=COLORS.raised,
        fg=COLORS.warning,
        activeforeground=COLORS.warning,
        highlightthickness=0,
        justify="left",
    ).pack(anchor="w", pady=(0, 8))
    grab_mode = tk.BooleanVar(value=False)
    grab_checkbox = tk.Checkbutton(
        sidebar,
        text="Click & grab",
        variable=grab_mode,
        command=lambda: on_grab_mode(grab_mode.get()),
        font=FONT_SMALL,
        selectcolor=COLORS.raised,
        highlightthickness=0,
    )
    if task_mode != "can_discard":
        grab_checkbox.pack(anchor="w", pady=(0, 6))
    ControlButton(
        sidebar,
        text="Discard can" if task_mode == "can_discard" else "Grab keys",
        command=on_grab_keys,
    ).pack(fill="x", pady=(0, 8))
    if task_mode == "can_discard":
        ControlButton(
            sidebar, text="Discard zone\ncalibration", command=on_discard_zone
        ).pack(fill="x", pady=(0, 8))
    ControlButton(
        sidebar,
        text="iMF rim grasp" if task_mode == "can_discard" else "iMF grasp",
        command=on_imf_grasp,
    ).pack(fill="x", pady=(0, 8))
    if controls:
        ControlButton(sidebar, text="Home", command=on_home).pack(fill="x", pady=(0, 8))
    from ..trajectory.path_overlays import COLORS as PATH_COLORS

    path_status = {}
    legend = tk.Frame(sidebar)
    legend.pack(fill="x", pady=(8, 0))
    for name, color in [*PATH_COLORS.items(), ("go", COLORS.path)]:
        row = tk.Frame(legend)
        row.pack(fill="x", pady=2)
        swatch = tk.Canvas(
            row, width=22, height=14, bg=COLORS.canvas, highlightthickness=0
        )
        swatch.create_line(1, 7, 19, 7, fill=color, width=3)
        swatch.pack(side="left", padx=(0, 6))
        variable = tk.StringVar(
            value="Go reference"
            if name == "go"
            else "iMF"
            if name == "imf"
            else "Direct"
            if name == "direct"
            else name.upper()
        )
        tk.Label(row, textvariable=variable, font=FONT_SMALL, fg=COLORS.text).pack(
            side="left"
        )
        if name != "go":
            path_status[name] = variable
    tk.Label(
        legend, text="× rejected · … updating", font=FONT_SMALL, fg=COLORS.muted
    ).pack(anchor="w", pady=(5, 0))

    camera_rail = tk.Frame(root)
    camera_rail.grid(
        row=2, column=2, rowspan=2, sticky="nsew", padx=(0, 8), pady=(0, 8)
    )
    camera_rail.grid_columnconfigure(0, weight=1)
    for row in range(3):
        camera_rail.grid_rowconfigure(row, weight=1, uniform="feeds", minsize=90)
    detail_strip = tk.Frame(camera_rail)
    detail_strip.grid(row=3, column=0, sticky="ew")
    zoom = tk.Label(detail_strip, width=64, height=64, bg=COLORS.raised, bd=0)
    zoom.pack(side="left", padx=(0, 6))
    depth_zoom = tk.Label(detail_strip, width=64, height=64, bg=COLORS.raised, bd=0)
    depth_zoom.pack(side="left", padx=(0, 6))
    for label in (zoom, depth_zoom):
        label.zoom_side = 64
    cursor_info = tk.StringVar()
    box = tk.Frame(detail_strip, width=155, height=80)
    box.pack(side="left", fill="x", expand=True)
    box.pack_propagate(False)
    tk.Label(
        box,
        textvariable=cursor_info,
        wraplength=148,
        justify="left",
        anchor="nw",
        font=FONT_SMALL,
    ).pack(fill="both", expand=True)

    def camera_panel(title, row, name):
        frame = panel(camera_rail, title)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        label = tk.Label(frame, name=name, bd=0, bg=COLORS.surface, width=1, height=1)
        label.pack(fill="both", expand=True)
        return label

    color = camera_panel("Color", 0, "color_view")
    hover = HoverControl(color.master, value=hover_m, on_change=on_hover)
    hover.pack(side="bottom", fill="x", padx=12, pady=(6, 10), before=color)
    depth = camera_panel("Depth", 1, "depth_view")
    webcam = camera_panel("Webcam", 2, "webcam_view")
    webcam.configure(text="Waiting for USB camera", fg=COLORS.muted)
    diagram_frame = panel(root, "3D workspace")
    diagram_frame.grid(row=2, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
    tk.Label(
        diagram_frame,
        text="Click: target · wheel: hover · Ctrl + drag / wheel: navigate",
        font=FONT_SMALL,
        fg=COLORS.muted,
        anchor="w",
        wraplength=400,
    ).pack(fill="x", padx=12, pady=(0, 6))
    diagram = tk.Label(
        diagram_frame, name="scene_view", bd=0, bg=COLORS.surface, width=1, height=1
    )
    diagram.pack(fill="both", expand=True)
    log_frame = panel(root, "Logs", width=280, height=180)
    log_frame.grid(row=3, column=1, sticky="nsew", padx=(0, 8), pady=(0, 8))
    log_frame.pack_propagate(False)
    fault_status = tk.StringVar(value="Waiting for joint feedback")
    fault_label = tk.Label(
        log_frame,
        textvariable=fault_status,
        bg=COLORS.surface,
        fg=COLORS.muted,
        anchor="nw",
        justify="left",
        font=FONT_SMALL,
    )
    fault_label.pack(fill="x", padx=12, pady=(0, 4))
    fault_label.bind(
        "<Configure>",
        lambda event: fault_label.configure(wraplength=max(80, event.width)),
    )
    log = EventLog(log_frame, height=4, width=20)
    log.pack(fill="both", expand=True, padx=12, pady=(0, 8))
    status = tk.StringVar(value=initial_status)
    status.trace_add("write", lambda *_: log.append(status.get()))
    log.append(initial_status)
    help_label = tk.Label(
        root,
        text="Close the jaws and capture 8 varied stationary tip poses."
        if calibration_mode
        else "",
        fg=COLORS.muted,
        font=FONT_SMALL,
        anchor="w",
    )
    help_label.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
    if not calibration_mode:
        help_label.grid_remove()
    return Dashboard(
        color,
        depth,
        diagram,
        zoom,
        depth_zoom,
        status,
        fault_status,
        cursor_info,
        torque_status,
        telemetry_status,
        go,
        help_label,
        fault_label,
        560,
        420,
        250,
        hover,
        disable,
        webcam,
        path_status,
    )
