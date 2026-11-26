"""Smooth preview-only height input, with high-resolution wheel support."""

import math
import time
import tkinter as tk

from .theme import COLORS, FONT_SMALL


def wheel_height(current, delta, minimum=0.02, maximum=0.20):
    return min(maximum, max(minimum, current + float(delta) / 120 * 0.002))


class HoverControl(tk.Frame):
    def __init__(self, parent, value=0.12, on_change=lambda _: None):
        super().__init__(parent, bg=COLORS.canvas)
        self.minimum = 0.02
        self.maximum = 0.20
        self.value = value
        self.display = value
        self.on_change = on_change
        self.pending = None
        self.last_tick = time.perf_counter()
        self.caption = tk.StringVar()
        self.grid_columnconfigure(0, weight=1)
        tk.Label(self, text="Hover", font=FONT_SMALL, fg=COLORS.muted).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(
            self,
            textvariable=self.caption,
            font=("Segoe UI", 12, "bold"),
            width=8,
            anchor="e",
        ).grid(row=0, column=1, sticky="e")
        self.track = tk.Canvas(
            self,
            width=228,
            height=16,
            bg=COLORS.canvas,
            highlightthickness=0,
            takefocus=True,
        )
        self.track.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        self.track.bind("<Button-1>", self._drag)
        self.track.bind("<B1-Motion>", self._drag)
        self.track.bind("<Left>", lambda _: self.adjust(-24))
        self.track.bind("<Right>", lambda _: self.adjust(24))
        self.track.bind("<Configure>", lambda _: self._paint())
        self.set_value(value, notify=False)

    def set_limits(self, minimum=0.02, maximum=0.20):
        self.minimum = minimum
        self.maximum = maximum
        self.set_value(self.value, notify=False)

    def set_value(self, value, notify=True):
        value = min(self.maximum, max(self.minimum, float(value)))
        changed = not math.isclose(value, self.value, abs_tol=1e-9)
        self.value = value
        self.caption.set(f"{value * 100:.2f} cm")
        if self.pending is None:
            self.last_tick = time.perf_counter()
            self._animate()
        if notify and changed:
            self.on_change(value)

    def adjust(self, delta):
        self.set_value(wheel_height(self.value, delta, self.minimum, self.maximum))
        return "break"

    def _drag(self, event):
        width = max(1, self.track.winfo_width() - 12)
        self.set_value(
            self.minimum
            + (self.maximum - self.minimum) * min(1, max(0, (event.x - 6) / width))
        )

    def _animate(self):
        now = time.perf_counter()
        dt = max(0.016, now - self.last_tick)
        self.last_tick = now
        self.display += (self.value - self.display) * (1 - math.exp(-dt / 0.06))
        if abs(self.display - self.value) < 0.00005:
            self.display = self.value
        self._paint()
        self.pending = (
            self.after(16, self._animate) if self.display != self.value else None
        )

    def _paint(self):
        self.track.delete("all")
        width = max(24, self.track.winfo_width())
        y = 8
        fraction = min(
            1,
            max(
                0,
                (self.display - self.minimum) / max(0.001, self.maximum - self.minimum),
            ),
        )
        x = 6 + (width - 12) * fraction
        self.track.create_line(6, y, width - 6, y, fill=COLORS.border, width=3)
        self.track.create_line(6, y, x, y, fill=COLORS.accent_light, width=3)
        self.track.create_oval(x - 4, y - 4, x + 4, y + 4, fill=COLORS.text, outline="")
