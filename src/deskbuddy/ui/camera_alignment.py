"""Camera-only A4 correspondence workflow; no robot or torque APIs."""

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..multiview import calibrate_views, paper_candidate
from ..overlays import project
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

        self.window.title("Align cameras · A4")

        self.window.geometry("1280x760")

        self.window.minsize(900, 600)

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.future, self.result = None, None

        self.version = 0

        self.dragging = None

        self.save_requested = False

        self.closed, self.pending = False, None

        self.labels, self.points = [], [[], []]

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

        if frames_provider is not None:
            ControlButton(tools, text="New snapshots", command=self.capture).pack(
                side="left", padx=(0, 6)
            )

        ControlButton(tools, text="Find paper", command=self.suggest).pack(side="left")

        for index, title in enumerate(["Reset RealSense", "Reset webcam"]):
            ControlButton(
                tools, text=title, command=lambda i=index: self.reset(i)
            ).pack(side="left", padx=6)

        tools2 = tk.Frame(self.window)

        tools2.pack(fill="x", padx=12, pady=(0, 10))

        ControlButton(
            tools2, text="Check alignment", command=self.check, primary=True
        ).pack(side="left", padx=6)

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

        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.suggest()

    def invalidate(self):

        self.version += 1

        self.result = None

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
            self.points[0] = bed.get("corner_pixels") or [
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

        self.future = self.executor.submit(
            calibrate_views,
            *self.images,
            *copy.deepcopy(self.points),
            self.intrinsics,
            self.config,
            self.depth,
            strict=False,
        )

        self.pending = self.window.after(75, self.finish)

    def finish(self):

        if self.closed:
            return

        if not self.future.done():
            self.pending = self.window.after(75, self.finish)

            return

        try:
            self.result = self.future.result()

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

            if self.save_requested:
                self.commit()

                return

        except (ValueError, RuntimeError) as exc:
            self.status.set(str(exc))

        self.future = None

    def commit(self):

        if self.result is not None:
            self.on_save(self.result)

            self.close()

        elif self.future is None:
            self.check()

            self.save_requested = self.future is not None

    def close(self):

        self.closed = True

        if self.pending is not None:
            self.window.after_cancel(self.pending)

        self.executor.shutdown(wait=False, cancel_futures=True)

        self.window.destroy()
