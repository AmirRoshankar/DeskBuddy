"""Camera-only A4 correspondence workflow; no robot or torque APIs."""

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..multiview import calibrate_views, paper_candidate
from ..overlays import project
from ..rgb_stereo_alignment import refine_rgb_pairs as check_elevated_tip
from .theme import COLORS, FONT_SMALL
from .viewport import CameraViewport
from .widgets import ControlButton


class CameraAlignment:
    def __init__(
        self,
        root,
        primary,
        secondary,
        depth,
        intrinsics,
        config,
        on_save,
        frames_provider=None,
    ):

        self.images = [primary, secondary]

        self.frames_provider = frames_provider

        self.depth, self.intrinsics, self.config, self.on_save = (
            depth,
            intrinsics,
            config,
            on_save,
        )

        self.window = tk.Toplevel(root)

        self.window.lift()

        self.window.title("Align cameras · A4")

        self.window.geometry("1280x760")

        self.window.minsize(800, 500)
        self.window.resizable(True, True)
        self.window.bind("<F11>", self.toggle_fullscreen)
        self.window.bind("<Escape>", lambda event: self.set_fullscreen(False))

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.future, self.result = None, None

        self.version = 0

        self.dragging = None

        self.save_requested = False

        self.closed, self.pending = False, None
        self.live_pending = None
        self.tip_depth = None

        self.labels, self.points = [], [[], []]
        self.tip_mode = False
        self.tip_pixels = [None, None]
        self.tip_observations = []

        self.status = tk.StringVar(
            value="Same physical corners in both views: 1 → 2 along the 21 cm edge, then around the sheet. Drag numbered corners to adjust. Blue markings check alignment."
        )

        tk.Label(
            self.window,
            textvariable=self.status,
            wraplength=1150,
            justify="left",
            font=FONT_SMALL,
            fg=COLORS.text,
        ).pack(fill="x", padx=12, pady=12)

        tools = tk.Frame(self.window)

        tools.pack(fill="x", padx=12, pady=(0, 10))

        ControlButton(tools, text="Find paper", command=self.suggest).pack(side="left")
        self.fullscreen_button = ControlButton(
            tools, text="Fullscreen", command=self.toggle_fullscreen
        )
        self.fullscreen_button.pack(side="right")

        tools2 = tk.Frame(self.window)

        tools2.pack(fill="x", padx=12, pady=(0, 10))

        ControlButton(
            tools2, text="Check alignment", command=self.check, primary=True
        ).pack(side="left", padx=6)

        ControlButton(tools2, text="Add matched point", command=self.begin_tip).pack(
            side="left", padx=6
        )

        ControlButton(tools2, text="Undo point", command=self.undo_tip).pack(
            side="left", padx=6
        )

        # Opt-in: the A4 on the level table fixes the RealSense tilt/height
        # when the hand-guided calibration leaves its corners >4 px off.
        self.level_camera = tk.BooleanVar(value=False)
        level_button = tk.Checkbutton(
            tools2,
            text="Correct RealSense tilt from A4",
            variable=self.level_camera,
            command=self.invalidate,
            font=FONT_SMALL,
            selectcolor=COLORS.raised,
            highlightthickness=0,
        )
        if not config.get("T_base_camera") or not config.get("calibration_verified"):
            level_button.configure(state="disabled")
        level_button.pack(side="left", padx=12)

        self.save = ControlButton(
            tools2, text="Save alignment", command=self.commit, state="disabled"
        )

        self.save.pack(side="right")

        panes = tk.Frame(self.window)

        panes.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        panes.grid_rowconfigure(1, weight=1)

        for i, title in enumerate(
            ["RealSense · reference", "Logitech · same numbered corners"]
        ):
            panes.grid_columnconfigure(i, weight=1, uniform="views")

            tk.Label(panes, text=title, font=FONT_SMALL).grid(
                row=0, column=i, sticky="w", pady=6
            )

            label = tk.Label(panes, width=1, height=1, bg=COLORS.surface)

            label.grid(row=1, column=i, sticky="nsew", padx=(0, 8))

            label.bind("<Button-1>", lambda event, i=i: self.click(i, event))

            label.bind("<B1-Motion>", lambda event, i=i: self.drag(i, event))

            label.bind("<ButtonRelease-1>", lambda event, i=i: self.release(i, event))

            label.bind("<Configure>", lambda event: self.paint())

            self.labels.append(label)

        self.window.configure(bg=COLORS.canvas)
        for container in self.window.winfo_children():
            if isinstance(container, tk.Frame):
                container.configure(bg=COLORS.canvas)
                for child in container.winfo_children():
                    if (
                        isinstance(child, (tk.Label, tk.Checkbutton))
                        and child not in self.labels
                    ):
                        child.configure(bg=COLORS.canvas, fg=COLORS.text)
            elif isinstance(container, tk.Label):
                container.configure(bg=COLORS.canvas, fg=COLORS.text)

        depth_frame = tk.Frame(self.window, bg=COLORS.canvas)
        depth_frame.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            depth_frame,
            text="Depth reference - RGB alignment does not use this",
            bg=COLORS.canvas,
            fg=COLORS.muted,
            font=FONT_SMALL,
        ).pack(side="left", padx=8)
        self.depth_label = tk.Label(depth_frame, bg=COLORS.surface)
        self.depth_label.pack(side="right")

        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.restore_or_suggest()
        self.refresh_live()
        self.window.after_idle(self.maximize)

    def restore_or_suggest(self):
        import copy

        candidates = [
            self.config.get(key)
            for key in ("secondary_camera_calibration", "camera_pair_alignment")
        ]
        candidates = [
            item
            for item in candidates
            if item
            and len(item.get("primary_pixels", [])) == 4
            and len(item.get("secondary_pixels", [])) == 4
        ]
        if not candidates:
            self.suggest()
            return
        saved = max(candidates, key=lambda item: item.get("calibrated_unix_s", 0))
        self.points = [
            copy.deepcopy(saved[key]) for key in ("primary_pixels", "secondary_pixels")
        ]
        self.result = copy.deepcopy(saved)
        self.tip_observations = copy.deepcopy(saved.get("matched_points", []))
        self.save.configure(state="normal")
        self.status.set(
            "Saved alignment loaded. Drag corners to adjust, or Find paper to replace them. "
            + saved.get("elevated_check_status", saved.get("pose_validation", ""))
        )
        if self.tip_observations and not self.result.get("metric_3d_validated"):
            self.update_tip_checks()
        self.paint()

    def maximize(self):
        if not self.closed:
            self.maximize_window()
            self.window.lift()
            self.window.focus_set()

    def maximize_window(self):
        import sys

        if sys.platform == "win32":
            self.window.state("zoomed")
        else:
            self.window.attributes("-zoomed", True)

    def toggle_fullscreen(self, event=None):
        self.set_fullscreen(not bool(self.window.attributes("-fullscreen")))
        return "break"

    def set_fullscreen(self, enabled):
        # Leave zoomed state first: Windows Tk can ignore fullscreen while zoomed.
        if enabled:
            self.window.state("normal")
        self.window.attributes("-fullscreen", enabled)
        if not enabled:
            self.maximize_window()
        self.fullscreen_button.configure(
            text="Exit fullscreen" if enabled else "Fullscreen"
        )
        self.window.lift()
        self.window.focus_set()

    def refresh_live(self):
        if self.closed:
            return
        try:
            frames = self.frames_provider() if self.frames_provider else None
            if frames is not None:
                self.images[0], self.images[1], self.depth = frames
                self.paint()
        except Exception as exc:
            self.status.set(f"Camera feed unavailable: {exc}")
        finally:
            if not self.closed and self.frames_provider:
                self.live_pending = self.window.after(66, self.refresh_live)

    def invalidate(self):

        self.version += 1

        self.result = None
        self.tip_mode = False

        self.save_requested = False

        self.save.configure(
            state="normal" if all(len(p) == 4 for p in self.points) else "disabled"
        )

    def capture(self):

        frames = self.frames_provider()

        if frames is None:
            self.status.set("Wait for fresh frames from both cameras.")

            return

        self.images[0], self.images[1], self.depth = frames

        self.suggest()

    def suggest(self):

        bed = self.config.get("test_bed")

        if bed:
            self.points[0] = [list(p) for p in bed.get("corner_pixels", [])] or [
                project(p, self.config["T_base_camera"], self.intrinsics)
                for p in bed["corners_base_m"]
            ]

            if any(p is None for p in self.points[0]):
                self.points[0] = []

        candidate = paper_candidate(self.images[1])

        self.points[1] = [] if candidate is None else candidate.tolist()

        self.invalidate()

        self.paint()

    def reset(self, index):

        self.points[index] = []

        self.invalidate()

        self.paint()

    def click(self, index, event):

        label = self.labels[index]

        pixel = label.viewport.to_camera(event.x, event.y)

        if pixel is None:
            return

        if self.tip_mode:
            self.tip_pixels[index] = list(pixel)
            if index == 0:
                self.tip_depth = None if self.depth is None else self.depth.copy()
            if all(p is not None for p in self.tip_pixels):
                self.tip_observations.append(
                    dict(
                        primary_pixel=self.tip_pixels[0],
                        secondary_pixel=self.tip_pixels[1],
                        depth=self.tip_depth,
                    )
                )
                self.tip_pixels = [None, None]
                self.tip_mode = False
                if self.result is None:
                    self.status.set(
                        f"{len(self.tip_observations)} pairs captured; A4 pose not checked yet. Press Check alignment to evaluate depth and camera error."
                    )
                else:
                    self.update_tip_checks()
            else:
                self.status.set(
                    "Click the same orange point in the other camera view; keep the arm still."
                )
            self.paint()
            return

        if len(self.points[index]) == 4:
            nearest = int(
                np.argmin(np.linalg.norm(np.array(self.points[index]) - pixel, axis=1))
            )

            self.points[index][nearest] = pixel

            self.dragging = (index, nearest)

        else:
            self.points[index].append(pixel)

            self.dragging = (index, len(self.points[index]) - 1)

        self.invalidate()

        self.paint()

    def drag(self, index, event):

        if self.dragging is None or self.dragging[0] != index:
            return

        pixel = self.labels[index].viewport.to_camera(event.x, event.y)

        if pixel is not None:
            self.points[index][self.dragging[1]] = pixel

            self.invalidate()

            self.paint()

    def release(self, index, event):

        self.drag(index, event)

        self.dragging = None

    def paint(self):
        if hasattr(self, "depth_label") and self.depth is not None:
            from ..views import depth_images

            preview = depth_images(self.depth)[0]
            preview.thumbnail((320, 140))
            self.depth_label.image = ImageTk.PhotoImage(preview)
            self.depth_label.configure(image=self.depth_label.image)

        for i, label in enumerate(self.labels):
            image = Image.fromarray(self.images[i]).copy()

            draw = ImageDraw.Draw(image)

            points = self.points[i]

            if len(points) == 4:
                draw.line(
                    [tuple(p) for p in points] + [tuple(points[0])],
                    fill="#68f1a6",
                    width=3,
                )

            for number, (x, y) in enumerate(points, 1):
                draw.ellipse(
                    (x - 7, y - 7, x + 7, y + 7),
                    fill="#ff9275",
                    outline="white",
                    width=2,
                )

                draw.text(
                    (x + 10, y - 15),
                    str(number),
                    fill="white",
                    stroke_width=2,
                    stroke_fill="#171719",
                )

            for number, observation in enumerate(self.tip_observations, 1):
                x, y = observation["primary_pixel" if i == 0 else "secondary_pixel"]
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#68f1a6", width=2)
                draw.text((x + 9, y), f"P{number}", fill="white")

            if self.tip_mode and self.tip_pixels[i] is not None:
                x, y = self.tip_pixels[i]
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#68f1a6", width=3)
                draw.text(
                    (x + 10, y), f"P{len(self.tip_observations) + 1}", fill="white"
                )

            label.viewport = CameraViewport(
                max(1, label.winfo_width()),
                max(1, label.winfo_height()),
                image.width,
                image.height,
            )

            label.image = ImageTk.PhotoImage(label.viewport.render(image))

            label.configure(image=label.image)

    def check(self):

        if self.future is not None:
            return

        if any(len(p) != 4 for p in self.points):
            self.status.set("Select four corresponding corners in each view.")

            return

        self.status.set(
            "Checking A4, blue markings and held-out RGB-D feature matches…"
        )

        import copy

        self.check_version = self.version

        config = dict(self.config, level_camera_to_paper=self.level_camera.get())
        self.future = self.executor.submit(
            calibrate_views,
            *[image.copy() for image in self.images],
            *copy.deepcopy(self.points),
            self.intrinsics,
            config,
            None if self.depth is None else self.depth.copy(),
            strict=False,
        )

        self.pending = self.window.after(75, self.finish)

    def begin_tip(self):
        self.dragging = None
        self.tip_mode = True
        self.tip_pixels = [None, None]
        self.tip_depth = None
        self.status.set(
            "Add point: click the same physical feature in each live view. Keep the feature still until both clicks are complete."
        )
        self.paint()

    def undo_tip(self):
        self.tip_mode = False
        self.tip_pixels = [None, None]
        if self.tip_observations:
            self.tip_observations.pop()
        self.update_tip_checks()
        self.paint()

    def update_tip_checks(self):
        if self.result is None:
            return
        saved = []
        for observation in self.tip_observations:
            item = {
                key: observation[key] for key in ("primary_pixel", "secondary_pixel")
            }
            if "depth_patch" in observation:
                item["depth_patch"] = observation["depth_patch"]
            elif observation.get("depth") is not None:
                u, v = map(lambda n: int(round(n)), observation["primary_pixel"])
                image = observation["depth"]
                if 2 <= u < image.shape[1] - 2 and 2 <= v < image.shape[0] - 2:
                    item["depth_patch"] = image[v - 2 : v + 3, u - 2 : u + 3].tolist()
            saved.append(item)
        self.result["matched_points"] = saved
        count = len(self.tip_observations)
        # Manual evidence is rechecked after every edit; no depth images are saved.
        self.result.pop("manual_elevated_check", None)
        self.result["metric_3d_validated"] = self.result.get(
            "automatic_metric_3d_validated", False
        )
        message = f"{count} point pairs captured."
        if self.result.get("camera_only"):
            message += (
                " Paper alignment only; robot-base pose is needed for metric 3D checks."
            )
        else:
            try:
                check = check_elevated_tip(
                    self.result, self.tip_observations, self.intrinsics, self.config
                )
                self.result["manual_elevated_check"] = check
                if check.get("candidate") and check["passed"]:
                    self.result.update(check["candidate"])
                self.result["metric_3d_validated"] = check["passed"]
                message = check.get(
                    "status",
                    f"{count} pairs: check passed"
                    if check["passed"]
                    else f"{count} pairs: check failed",
                )
                self.result["pose_validation"] = message
            except (ValueError, TypeError) as exc:
                message = f"3D check: {exc}"
        self.result["elevated_check_status"] = message
        if not self.result["metric_3d_validated"]:
            self.result["pose_validation"] = message
        self.status.set(message)

    def finish(self):

        if self.closed:
            return

        if not self.future.done():
            self.pending = self.window.after(75, self.finish)

            return

        try:
            self.result = self.future.result()
            self.result["automatic_metric_3d_validated"] = self.result.get(
                "metric_3d_validated", False
            )

            if self.check_version != self.version:
                self.result = None

                self.future = None

                self.status.set(
                    "Corners changed during the check · Check alignment again."
                )

                return

            blue = self.result.get("blue_agreement")

            blue_text = (
                "blue markings unavailable"
                if blue is None
                else f"blue agreement {blue:.0%}"
            )
            refinement = self.result.get("primary_camera_refinement")
            if refinement:
                blue_text += (
                    f" · RealSense tilt corrected {refinement['rotation_change_deg']:.1f}°"
                    f" (corners now ≤{max(refinement['corner_residual_px']):.1f} px)"
                )

            self.status.set(
                f"A4 aligned · {blue_text} · {self.result.get('pose_validation', 'table-plane alignment only')}. Review the numbered corners, then Save alignment."
            )

            warnings = self.result.get("validation_warnings", [])

            if warnings:
                self.status.set(
                    "Check warning · "
                    + "; ".join(warnings)
                    + ". Save alignment is available."
                )

            self.save.configure(state="normal")
            if self.tip_observations:
                self.update_tip_checks()

            if self.save_requested:
                self.commit()

                return

        except Exception as exc:
            self.status.set(f"Alignment check failed: {type(exc).__name__}: {exc}")

        self.future = None

    def commit(self):

        if self.result is not None:
            try:
                self.on_save(self.result)
            except Exception as exc:
                self.status.set(f"Alignment was not saved: {type(exc).__name__}: {exc}")
                return
            self.close()

        elif self.future is None:
            self.check()

            self.save_requested = self.future is not None

    def close(self):

        self.closed = True

        if self.pending is not None:
            self.window.after_cancel(self.pending)

        if self.live_pending is not None:
            self.window.after_cancel(self.live_pending)

        self.executor.shutdown(wait=False, cancel_futures=True)

        self.window.destroy()
