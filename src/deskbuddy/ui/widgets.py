"""Reusable presentation widgets with explicit hover, focus and disabled states."""

import time
import tkinter as tk

from .theme import COLORS, FONT_CONTROL, FONT_DATA, FONT_SMALL


class ControlButton(tk.Button):
    def __init__(self, parent, *, primary=False, tone=None, **kwargs):
        self.primary = primary
        self.tone = tone
        super().__init__(
            parent,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS.border,
            highlightcolor=COLORS.accent_light,
            font=FONT_CONTROL,
            padx=18,
            pady=9,
            cursor="hand2",
            takefocus=True,
            fg="white",
            disabledforeground="white",
            activeforeground="white",
            **kwargs,
        )
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", lambda _: self._rest())
        self._rest()

    def _rest(self):
        disabled = str(self.cget("state")) == "disabled"
        if self.tone in ("go", "stop"):
            background = (
                (COLORS.go_disabled if disabled else COLORS.go)
                if self.tone == "go"
                else COLORS.stop
            )
            super().configure(bg=background, activebackground=self._hover_color())
            return
        background = (
            COLORS.disabled
            if disabled and self.primary
            else (COLORS.accent if self.primary else COLORS.raised)
        )
        super().configure(
            bg=background,
            activebackground=COLORS.accent_hover if self.primary else COLORS.border,
        )

    def _enter(self, _):
        if str(self.cget("state")) != "disabled":
            super().configure(bg=self._hover_color())

    def _hover_color(self):
        if self.tone == "go":
            return COLORS.go_hover
        if self.tone == "stop":
            return COLORS.stop_hover
        return COLORS.accent_hover if self.primary else COLORS.border

    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        if "state" in kwargs:
            self._rest()
        return result


def panel(parent, title, detail="", **kwargs):
    frame = tk.Frame(
        parent,
        bg=COLORS.surface,
        highlightthickness=1,
        highlightbackground=COLORS.border,
        **kwargs,
    )
    heading = tk.Frame(frame, bg=COLORS.surface)
    heading.pack(fill="x", padx=12, pady=(9, 7))
    tk.Label(
        heading,
        text=title,
        bg=COLORS.surface,
        fg=COLORS.text,
        font=("Segoe UI", 10, "bold"),
    ).pack(side="left")
    if detail:
        tk.Label(
            heading, text=detail, bg=COLORS.surface, fg=COLORS.muted, font=FONT_SMALL
        ).pack(side="right")
    return frame


class EventLog(tk.Text):
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=COLORS.surface,
            fg=COLORS.muted,
            insertbackground=COLORS.text,
            selectbackground=COLORS.accent,
            selectforeground=COLORS.text,
            wrap="word",
            font=FONT_DATA,
            bd=0,
            highlightthickness=0,
            state="disabled",
            **kwargs,
        )
        self.last_message = None

    def append(self, message):
        if not message or message == self.last_message:
            return
        self.last_message = message
        self.configure(state="normal")
        self.insert("end", time.strftime("%H:%M:%S") + "  " + message + "\n")
        if int(self.index("end-1c").split(".")[0]) > 250:
            self.delete("1.0", "30.0")
        self.see("end")
        self.configure(state="disabled")
