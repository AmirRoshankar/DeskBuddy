"""Live camera comparison previews. This window has no motion API."""

import copy
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..overlays import project
from ..test_bed import surface_point, table_height_at
from ..trajectory.perception import draw_boxes, grouped_footprints
from .hover import HoverControl
from .theme import COLORS, FONT_DATA, FONT_SMALL
from .viewport import CameraViewport
from .widgets import ControlButton


def full_arm_results(results, arm, q, target, config, boxes, candidates):
    """Apply the same physical-space validation to every raw learned proposal."""
    from ..trajectory.execution import lift_route

    for result in results:
        began = time.perf_counter()
        result["xy_valid_candidates"] = result["valid_candidates"]
        valid, failures = [], []
        for path in result["raw_paths"]:
            try:
                plan = lift_route(
                    arm,
                    q,
                    target,
                    np.array(path),
                    config,
                    method=result["method"],
                    candidates=candidates,
                    boxes=boxes,
                    scene_revision=0,
                )
                valid.append((path, plan))
                failures.append(None)
            except ValueError as exc:
                failures.append(str(exc))
        result["valid_candidates"] = len(valid)
        result["full_arm_rejections"] = failures
        result["selected_path"] = None
        result["selected_tip_path_m"] = None
        if valid:
            path, plan = min(valid, key=lambda item: item[1].time_s[-1])
            result["selected_path"] = path
            result["selected_tip_path_m"] = [
                arm.fk(p).tolist() for p in plan.joints_rad
            ]
        result["joint_validation_ms"] = (time.perf_counter() - began) * 1000
        result["total_ms"] += result["joint_validation_ms"]
    from ..trajectory.joint_planner import plan_joint_route

    began = time.perf_counter()
    row = dict(
        method="rrt",
        candidates=1,
        valid_candidates=0,
        raw_paths=[],
        selected_path=None,
        selected_tip_path_m=None,
        full_arm_rejections=[],
    )
    try:
        plan = plan_joint_route(arm, q, target, config, boxes, candidates, 0)
        points = [arm.fk(p).tolist() for p in plan.joints_rad]
        row.update(
            valid_candidates=1,
            selected_path=[p[:2] for p in points],
            selected_tip_path_m=points,
            raw_paths=[[p[:2] for p in points]],
        )
    except ValueError as exc:
        row["full_arm_rejections"] = [str(exc)]
    row["total_ms"] = (time.perf_counter() - began) * 1000
    row["planning_space"] = "joint"
    results.append(row)
    return results


class ComparisonWindow:
    def __init__(self, root, provider, on_close):
        self.provider, self.on_close = provider, on_close
        self.window = tk.Toplevel(root)
        self.window.title("Trajectory comparison")
        self.window.geometry("1180x760")
        self.window.minsize(900, 620)
        self.window.configure(bg=COLORS.canvas)
        self.window.grid_columnconfigure(0, weight=3, uniform="comparison")
        self.window.grid_columnconfigure(1, weight=2, uniform="comparison")
        self.window.grid_rowconfigure(1, weight=1)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.suite = None
        self.future = None
        self.results = []
        self.goal = None
        self.surface = None
        self.seed = 1729
        self.closed = False
        self.snapshot = None
        self.result_boxes = None
        self.result_start = None
        self.request_pose = None
        self.version = 0
        self.request_geometry = None
        tools = tk.Frame(self.window, bg=COLORS.canvas)
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        ControlButton(
            tools, text="Compare paths", command=self.compare, primary=True
        ).pack(side="left")
        self.count = tk.IntVar(value=1)
        for k in (1, 4):
            tk.Radiobutton(
                tools,
                text=f"{k} candidate{'s' if k > 1 else ''}",
                variable=self.count,
                value=k,
                command=lambda: self.clear_results(
                    "Candidate count changed · Compare paths again."
                ),
                bg=COLORS.canvas,
                fg=COLORS.text,
                selectcolor=COLORS.raised,
                activebackground=COLORS.canvas,
                activeforeground=COLORS.text,
                highlightbackground=COLORS.canvas,
                highlightcolor=COLORS.accent_light,
            ).pack(side="left", padx=8)
        tk.Label(
            tools,
            text="Click camera to set target · preview only · no motor commands",
            fg=COLORS.muted,
            font=FONT_SMALL,
        ).pack(side="right")
        self.image = tk.Label(self.window, bg=COLORS.surface, width=1, height=1)
        self.image.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(0, 12))
        self.image.bind("<Button-1>", self.click)
        self.image.bind("<MouseWheel>", lambda e: self.hover.adjust(e.delta))
        self.image.bind("<Button-4>", lambda e: self.hover.adjust(120))
        self.image.bind("<Button-5>", lambda e: self.hover.adjust(-120))
        side = tk.Frame(self.window, bg=COLORS.surface, padx=12, pady=12)
        side.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(0, 12))
        self.hover = HoverControl(side, value=0.06, on_change=self.change_height)
        self.hover.pack(fill="x", pady=(0, 12))
        self.height_initialized = False
        self.dimensions = tk.StringVar()
        self.message = tk.StringVar(
            value="Choose a target clear of the orange boxes, then Compare paths."
        )
        dimension_label = tk.Label(
            side,
            textvariable=self.dimensions,
            bg=COLORS.surface,
            fg=COLORS.text,
            font=FONT_DATA,
            justify="left",
            anchor="nw",
            wraplength=380,
            width=1,
        )
        dimension_label.pack(fill="x", pady=(0, 20))
        self.method = tk.StringVar(value="imf")
        table = tk.Frame(side, bg=COLORS.surface)
        table.pack(fill="x")
        for col, text in enumerate(("Planner", "Valid", "Time ms")):
            tk.Label(
                table, text=text, bg=COLORS.surface, fg=COLORS.muted, font=FONT_SMALL
            ).grid(row=0, column=col, sticky="w", padx=5, pady=6)
        self.cells = {}
        for row, (name, title) in enumerate(
            (
                ("mlp", "MLP"),
                ("fm", "FM · 8 NFE"),
                ("imf", "iMF · 1 NFE"),
                ("rrt", "RRT-Connect"),
            ),
            1,
        ):
            tk.Radiobutton(
                table,
                text=title,
                variable=self.method,
                value=name,
                bg=COLORS.surface,
                fg=COLORS.text,
                selectcolor=COLORS.raised,
                activebackground=COLORS.surface,
                activeforeground=COLORS.text,
                highlightbackground=COLORS.surface,
                highlightcolor=COLORS.accent_light,
            ).grid(row=row, column=0, sticky="w", pady=5)
            self.cells[name] = []
            for col in (1, 2):
                label = tk.Label(
                    table, text="—", bg=COLORS.surface, fg=COLORS.muted, font=FONT_DATA
                )
                label.grid(row=row, column=col, sticky="w", padx=8)
                self.cells[name].append(label)
        legend = tk.Label(
            side,
            text="Same scene and arm pose for all planners.\nValid = whole arm, table and objects checked at the selected height.\nTime includes joint lifting and validation.\nRRT is separate; no hidden fallback.\nDashed outlines show the older XY model's 18 mm proxy.",
            bg=COLORS.surface,
            fg=COLORS.muted,
            font=FONT_SMALL,
            justify="left",
            wraplength=380,
            width=1,
            anchor="nw",
        )
        legend.pack(fill="x", pady=20)
        message_label = tk.Label(
            side,
            textvariable=self.message,
            bg=COLORS.surface,
            fg=COLORS.warning,
            justify="left",
            wraplength=380,
            width=1,
            anchor="nw",
        )
        message_label.pack(fill="x")
        for label in (dimension_label, legend, message_label):
            label.bind(
                "<Configure>",
                lambda event, widget=label: widget.configure(
                    wraplength=max(80, event.width)
                ),
            )
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def click(self, event):
        pixel = self.viewport.to_camera(event.x, event.y)
        if pixel is None or self.snapshot is None:
            return
        _, _, intrinsics, config, _, _ = self.snapshot
        try:
            point = surface_point(*pixel, intrinsics, config)
            low, high = (
                np.array(config["workspace_min_m"]),
                np.array(config["workspace_max_m"]),
            )
            if ((point[:2] < low[:2]) | (point[:2] > high[:2])).any():
                raise ValueError("Target outside playground XY")
            self.goal = point[:2]
            self.surface = point.copy()
            self.update_height_limits(config, intrinsics)
            self.clear_results("Target selected. Compare paths.")
        except ValueError as exc:
            self.message.set(str(exc))

    def update_height_limits(self, config, intrinsics):
        from ..overlays import visible_hover_bounds
        from ..planning import hover_bounds

        limits = hover_bounds(self.surface, config)
        rgb = self.snapshot[0]
        limits = visible_hover_bounds(
            self.surface,
            limits,
            config["T_base_camera"],
            intrinsics,
            (rgb.shape[1], rgb.shape[0]),
        )
        self.hover.set_limits(*limits)

    def change_height(self, value):
        self.clear_results("Height changed · Compare paths again.")

    def compare(self):
        if self.future is not None or self.snapshot is None:
            return
        rgb, _, intrinsics, config, q, boxes = self.snapshot
        config = copy.deepcopy(config)
        config["allow_object_collisions"] = False
        if self.goal is None:
            self.message.set("Click a target in the camera first.")
            return
        start = self.provider.arm.fk(q)[:2]
        goal = self.goal.copy()
        count = self.count.get()
        height = self.hover.value
        surface = self.surface.copy()
        seed = self.seed
        self.seed += 1
        captured = [b.as_dict() for b in boxes]
        version = self.version
        self.request_geometry = np.r_[
            config["workspace_min_m"],
            config["workspace_max_m"],
            np.asarray(config["T_base_camera"]).ravel(),
        ]
        self.request_pose = np.asarray(q).copy()
        self.message.set("Comparing all planners…")

        def work():
            from ..trajectory.live_preview import (
                PlannerSuite,
                preview_scene,
                save_preview,
            )

            scene = preview_scene(config, boxes)
            if self.suite is None:
                self.suite = PlannerSuite()
            results = [
                r
                for r in self.suite.compare(scene, start, goal, count, seed)
                if r["method"] != "rrt"
            ]
            target = surface + [0, 0, height]
            results = full_arm_results(
                results, self.provider.arm, q, target, config, boxes, count
            )
            payload = dict(
                unix_time_s=time.time(),
                preview_only=True,
                hardware_executable=False,
                scene=asdict(scene),
                start_m=start.tolist(),
                goal_m=goal.tolist(),
                seed=seed,
                detected_boxes=captured,
                results=results,
                hover_m=height,
                goal_3d_m=target.tolist(),
                start_joints_rad=np.asarray(q).tolist(),
                validity="full arm, physical table, self collision and detected objects; no repair/fallback",
                perception="YOLOE-26 Small RGB masks + filtered depth; paper and robot excluded",
            )
            folder = save_preview(payload)
            return results, captured, start, folder, version

        self.future = self.executor.submit(work)

    def refresh(self):
        if self.closed:
            return
        try:
            self.snapshot = self.provider()
            rgb, _, intrinsics, config, q, boxes = self.snapshot
            if not self.height_initialized:
                self.hover.set_value(config["hover_m"], notify=False)
                self.height_initialized = True
            text = ["RGB objects · estimated dimensions"]
            for index, box in enumerate(boxes, 1):
                d = (box.high - box.low) * 100
                text.append(f"Box {index}: {d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} cm")
            if not boxes:
                text.append("No raised objects detected")
            if len(boxes) > 2:
                text.append(
                    f"{len(boxes)} objects grouped into 2 conservative regions for all planners"
                )
            processing = getattr(self.provider, "perception_ms", None)
            if processing is not None:
                text.append(f"YOLO + 3D estimation: {processing:.1f} ms")
            self.dimensions.set("\n".join(text))
            if self.future is not None and self.future.done():
                try:
                    (
                        self.results,
                        self.result_boxes,
                        self.result_start,
                        folder,
                        version,
                    ) = self.future.result()
                    if version != self.version:
                        self.results = []
                    for result in self.results:
                        cells = self.cells[result["method"]]
                        cells[0].configure(
                            text=f"{result['valid_candidates']}/{result['candidates']}",
                            fg=COLORS.text
                            if result["valid_candidates"]
                            else COLORS.warning,
                        )
                        cells[1].configure(text=f"{result['total_ms']:.2f}")
                    self.message.set(
                        "Compared. Select a planner row to view its path. Saved local trial."
                        if self.results
                        else "Input changed · Compare paths again."
                    )
                except Exception as exc:
                    self.message.set(f"Preview unavailable: {exc}")
                self.future = None
            if self.results and self.result_boxes is not None:
                changed = len(boxes) != len(self.result_boxes)
                if not changed:
                    changed = any(
                        np.max(
                            np.abs(
                                np.r_[b.low, b.high] - np.r_[r["low_m"], r["high_m"]]
                            )
                        )
                        > 0.015
                        for b, r in zip(boxes, self.result_boxes)
                    )
                changed |= (
                    np.linalg.norm(self.provider.arm.fk(q)[:2] - self.result_start)
                    > 0.01
                )
                changed |= np.max(
                    np.abs(np.asarray(q) - self.request_pose)
                ) > np.deg2rad(2)
                if changed:
                    self.clear_results("Scene or start changed · Compare paths again.")
                geometry = np.r_[
                    config["workspace_min_m"],
                    config["workspace_max_m"],
                    np.asarray(config["T_base_camera"]).ravel(),
                ]
                if not np.allclose(geometry, self.request_geometry, atol=1e-8, rtol=0):
                    self.clear_results(
                        "Bounds or calibration changed · Compare paths again."
                    )
            image = Image.fromarray(rgb).copy()
            draw_boxes(image, boxes, config, intrinsics)
            draw = ImageDraw.Draw(image)

            def pixel(xy):
                margin = (
                    config.get("test_bed", {}).get("clearance_margin_m", 0.003)
                    if config.get("test_bed")
                    else 0
                )
                point = np.r_[
                    xy, float(table_height_at(xy, config)) - margin + self.hover.value
                ]
                return project(point, config["T_base_camera"], intrinsics)

            from ..motion_overlay import dashed

            for rectangle in grouped_footprints(boxes):
                low, high = rectangle[:2] - 0.018, rectangle[2:] + 0.018
                vertices = [
                    pixel(p)
                    for p in (
                        [low[0], low[1]],
                        [high[0], low[1]],
                        [high[0], high[1]],
                        [low[0], high[1]],
                    )
                ]
                for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                    if a is not None and b is not None:
                        dashed(draw, a, b, "#ffba75", width=1, dash=3)
            for result in self.results:
                if result["method"] != self.method.get():
                    continue
                path = result["selected_path"]
                valid = path is not None
                if path is None and result["raw_paths"]:
                    path = result["raw_paths"][0]
                if path is not None:
                    points = (
                        [
                            project(p, config["T_base_camera"], intrinsics)
                            for p in result["selected_tip_path_m"]
                        ]
                        if result.get("selected_tip_path_m")
                        else [pixel(p) for p in path]
                    )
                    for a, b in zip(points[:-1], points[1:]):
                        if a is not None and b is not None:
                            draw.line(
                                (a, b), fill="#77cbe8" if valid else "#ff9275", width=3
                            )
            if self.goal is not None:
                anchor = project(self.surface, config["T_base_camera"], intrinsics)
                p = pixel(self.goal)
                if p is not None:
                    if anchor is not None:
                        dashed(draw, anchor, p, "#68f1a6", width=1, dash=3)
                        draw.ellipse(
                            (
                                anchor[0] - 4,
                                anchor[1] - 4,
                                anchor[0] + 4,
                                anchor[1] + 4,
                            ),
                            fill="#68f1a6",
                        )
                    draw.ellipse(
                        (p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7),
                        outline="#68f1a6",
                        width=2,
                    )
            w, h = max(1, self.image.winfo_width()), max(1, self.image.winfo_height())
            self.viewport = CameraViewport(w, h, image.width, image.height)
            self.image.image = ImageTk.PhotoImage(self.viewport.render(image))
            self.image.configure(image=self.image.image)
        except (ValueError, RuntimeError, OSError) as exc:
            self.message.set(str(exc))
        self.window.after(150, self.refresh)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()
        self.on_close()

    def clear_results(self, message):
        self.version += 1
        self.results = []
        for cells in self.cells.values():
            for cell in cells:
                cell.configure(text="—", fg=COLORS.muted)
        self.message.set(message)
