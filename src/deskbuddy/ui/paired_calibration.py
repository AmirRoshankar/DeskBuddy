"""Live, hand-guided RGB tip capture. No motor commands."""

import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .theme import COLORS, FONT_SMALL
from .viewport import CameraViewport


class PairedCalibration:
    def __init__(self, root, frames, capture_state, save_pair):
        self.frames, self.capture_state, self.save_pair = (
            frames,
            capture_state,
            save_pair,
        )
        self.window = tk.Toplevel(root)
        self.window.title("Manual calibration - 8 tip pairs")
        self.window.geometry("1280x800")
        self.window.configure(bg=COLORS.canvas)
        self.window.resizable(True, True)
        self.count, self.pixels, self.context = 0, [None, None], None
        self.closed, self.pending = False, None
        self.status = tk.StringVar()
        tk.Label(
            self.window,
            textvariable=self.status,
            bg=COLORS.canvas,
            fg=COLORS.text,
            font=FONT_SMALL,
            wraplength=1200,
        ).pack(fill="x", pady=12)
        self.labels = []
        panes = tk.Frame(self.window, bg=COLORS.canvas)
        panes.pack(fill="both", expand=True)
        panes.grid_rowconfigure(1, weight=1)
        for i, title in enumerate(("RealSense RGB", "Webcam RGB")):
            panes.grid_columnconfigure(i, weight=1, uniform="cameras")
            tk.Label(panes, text=title, bg=COLORS.canvas, fg=COLORS.text).grid(
                row=0, column=i
            )
            label = tk.Label(panes, bg=COLORS.surface)
            label.grid(row=1, column=i, sticky="nsew", padx=6, pady=6)
            label.bind("<Button-1>", lambda event, index=i: self.click(index, event))
            self.labels.append(label)
        self.window.bind(
            "<F11>",
            lambda event: self.window.attributes(
                "-fullscreen", not self.window.attributes("-fullscreen")
            ),
        )
        self.window.bind(
            "<Escape>", lambda event: self.window.attributes("-fullscreen", False)
        )
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.prompt()
        self.refresh()
        self.window.after_idle(lambda: self.window.state("zoomed"))

    def prompt(self):
        self.status.set(
            f"Pose {self.count + 1}/8 - click the jaw tip in BOTH views. Keep the arm still between the two clicks. After each pair, hand-guide to a different position and height."
        )

    def refresh(self):
        if self.closed:
            return
        try:
            frames = self.frames()
            if frames is not None:
                self.images = frames[:2]
                for i, label in enumerate(self.labels):
                    image = Image.fromarray(self.images[i])
                    if self.pixels[i] is not None:
                        x, y = self.pixels[i]
                        ImageDraw.Draw(image).ellipse(
                            (x - 7, y - 7, x + 7, y + 7), outline="#68f1a6", width=3
                        )
                    label.viewport = CameraViewport(
                        max(1, label.winfo_width()),
                        max(1, label.winfo_height()),
                        image.width,
                        image.height,
                    )
                    label.image = ImageTk.PhotoImage(label.viewport.render(image))
                    label.configure(image=label.image)
        except Exception as exc:
            self.status.set(f"Camera update: {exc}")
        self.pending = self.window.after(66, self.refresh)

    def click(self, index, event):
        label = self.labels[index]
        if not hasattr(label, "viewport"):
            return
        pixel = label.viewport.to_camera(event.x, event.y)
        if pixel is None:
            return
        try:
            current = self.capture_state()
            if self.context is None:
                self.context = current
            elif np.max(
                np.abs(np.asarray(current["q_model_rad"]) - self.context["q_model_rad"])
            ) > np.deg2rad(0.7):
                raise ValueError(
                    "Arm moved between clicks; hold still and select both tips again"
                )
            self.pixels[index] = list(pixel)
            if all(point is not None for point in self.pixels):
                self.save_pair(self.pixels, self.context, self.images)
                self.count += 1
                self.pixels, self.context = [None, None], None
                if self.count == 8:
                    self.close()
                    return
                self.prompt()
            else:
                self.status.set(
                    f"Pose {self.count + 1}/8 - {self.count * 2 + 1}/16 clicks. Click the same jaw tip in the other view; keep the arm still."
                )
        except (ValueError, RuntimeError) as exc:
            self.pixels, self.context = [None, None], None
            self.status.set(f"Pair not captured: {exc}")

    def close(self):
        self.closed = True
        if self.pending is not None:
            self.window.after_cancel(self.pending)
        self.window.destroy()
