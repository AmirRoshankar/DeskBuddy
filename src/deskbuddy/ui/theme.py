"""Semantic design tokens shared by native widgets and rendered diagrams."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    canvas: str = "#171719"
    surface: str = "#202023"
    raised: str = "#29292D"
    border: str = "#3C3C42"
    text: str = "#F5F3F0"
    muted: str = "#B1AFAD"
    accent: str = "#C2472D"
    accent_hover: str = "#A93C26"
    accent_light: str = "#FF9275"
    disabled: str = "#55423D"
    warning: str = "#F4B88C"
    go: str = "#246B49"
    go_hover: str = "#1B5539"
    go_disabled: str = "#344B3F"
    stop: str = "#AD3D35"
    stop_hover: str = "#8D302B"
    path: str = "#F46D67"


COLORS = Palette()
FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_CONTROL = ("Segoe UI", 10, "bold")
FONT_DATA = ("Consolas", 9)


def apply(root):
    root.configure(bg=COLORS.canvas)
    root.option_add("*Font", FONT)
    root.option_add("*Background", COLORS.canvas)
    root.option_add("*Foreground", COLORS.text)
    root.option_add("*selectBackground", COLORS.accent)
    root.option_add("*selectForeground", COLORS.text)
    root.option_add("*insertBackground", COLORS.text)
