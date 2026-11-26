"""Inline calibration guidance; hardware operations remain in the controller."""

import tkinter as tk

from .theme import COLORS, FONT_SMALL
from .widgets import ControlButton


class CalibrationBar(tk.Frame):
    def __init__(self, root):
        super().__init__(root, bg=COLORS.surface, padx=12, pady=8)
        self.message = tk.StringVar()
        self.label = tk.Label(
            self,
            textvariable=self.message,
            justify="left",
            anchor="w",
            bg=COLORS.surface,
            fg=COLORS.text,
            font=FONT_SMALL,
            width=1,
        )
        self.label.pack(side="left", fill="both", expand=True)
        self.action = ControlButton(self, text="", command=lambda: None)
        self.action.pack(side="right", padx=(12, 0))
        self.retry = ControlButton(self, text="Retry corners", command=lambda: None)
        self.label.bind(
            "<Configure>", lambda e: self.label.configure(wraplength=max(120, e.width))
        )

    def step(self, message, label="", action=lambda: None, enabled=True, retry=None):
        self.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        self.message.set(message)
        self.retry.pack_forget()
        if retry is not None:
            self.retry.configure(command=retry)
            self.retry.pack(side="right", padx=(12, 0))
        self.action.configure(
            text=label, command=action, state="normal" if enabled else "disabled"
        )
        if label:
            self.action.pack(side="right", padx=(12, 0))
        else:
            self.action.pack_forget()
