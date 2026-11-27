"""One-click fixed-jaw feature selection from a live webcam; no robot API."""

import tkinter as tk

from PIL import Image, ImageTk

from .theme import COLORS
from .viewport import CameraViewport


class TipPicker:
    def __init__(self, root, frames, on_select):
        self.frames, self.on_select = frames, on_select
        self.window = tk.Toplevel(root)
        self.window.title("Select the fixed jaw tip")
        self.window.geometry("1100x760")
        self.window.configure(bg=COLORS.canvas)
        self.closed, self.pending = False, None
        tk.Label(
            self.window,
            text="Click the contact tip of the FIXED jaw once. Choose a visible edge/detail; do not select the can or painted background.",
            bg=COLORS.canvas,
            fg=COLORS.text,
            wraplength=1000,
        ).pack(fill="x", pady=10)
        self.label = tk.Label(self.window, bg=COLORS.surface)
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self.click)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def refresh(self):
        if self.closed:
            return
        frame = self.frames()
        if frame is not None:
            self.image, self.stamp = frame
            image = Image.fromarray(self.image)
            self.viewport = CameraViewport(
                max(1, self.label.winfo_width()),
                max(1, self.label.winfo_height()),
                image.width,
                image.height,
            )
            self.label.image = ImageTk.PhotoImage(self.viewport.render(image))
            self.label.configure(image=self.label.image)
        self.pending = self.window.after(50, self.refresh)

    def click(self, event):
        if not hasattr(self, "viewport"):
            return
        pixel = self.viewport.to_camera(event.x, event.y)
        if pixel is not None:
            self.on_select(self.image.copy(), pixel, self.stamp)
            self.close()

    def close(self):
        self.closed = True
        if self.pending is not None:
            self.window.after_cancel(self.pending)
        self.window.destroy()
