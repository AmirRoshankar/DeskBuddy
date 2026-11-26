"""Inspectable iMF proposals; no motor calls or implicit default-policy fallback."""

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .theme import COLORS, FONT_SMALL
from .widgets import ControlButton


class GraspCandidates:
    def __init__(self, root, provider=None):
        self.provider = provider
        self.window = tk.Toplevel(root)
        self.window.title("iMF grasp candidates")
        self.window.geometry("680x660")
        self.window.minsize(600, 620)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.pending = None
        self.closed = False
        self.seed = 1739
        self.status = tk.StringVar(
            value="Synthetic box model · 1 network evaluation · no motor execution"
        )
        tk.Label(
            self.window,
            textvariable=self.status,
            font=FONT_SMALL,
            wraplength=640,
            justify="left",
        ).pack(fill="x", padx=16, pady=12)
        self.target_info = tk.StringVar(
            value="Detecting the target from the shared camera scene…"
        )
        tk.Label(
            self.window,
            textvariable=self.target_info,
            font=FONT_SMALL,
            wraplength=640,
            justify="left",
        ).pack(fill="x", padx=16)
        ControlButton(
            self.window,
            text="Refresh automatic grasp",
            command=self.sample,
            primary=True,
        ).pack(anchor="w", padx=16, pady=12)
        self.plot = tk.Label(self.window, bg=COLORS.surface)
        self.plot.pack(fill="both", expand=True, padx=16)
        self.details = tk.StringVar(value="")
        tk.Label(
            self.window,
            textvariable=self.details,
            font=FONT_SMALL,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=16, pady=8)
        tk.Label(
            self.window,
            text="Execution needs the open-jaw center → fixed-jaw tip offset.\nThese geometric candidates are not measured pickup successes.",
            font=FONT_SMALL,
            fg=COLORS.warning,
            justify="left",
        ).pack(fill="x", padx=16, pady=(0, 12))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.sample()

    def sample(self):
        if self.future is not None:
            return
        if self.provider is None:
            self.status.set("Scene-aware iMF needs the live camera scene.")
            return
        from ..learned.grasp_proposals import propose_scene

        self.status.set("Detecting target and ranking 32 iMF candidates…")
        self.future = self.executor.submit(propose_scene, self.provider())
        self.pending = self.window.after(80, self.finish)

    def finish(self):
        if self.closed:
            return
        if not self.future.done():
            self.pending = self.window.after(80, self.finish)
            return
        try:
            result = self.future.result()
            entries = result["candidates"]
            self.entries = entries
            self.size = np.array(result["size_m"])
            dims = self.size * 1000
            self.target_info.set(
                f"{result['source']} · {dims[0]:.0f} × {dims[1]:.0f} × {dims[2]:.0f} mm · best valid candidate selected automatically"
            )
            self.status.set(
                f"iMF · {sum(e['geometric_valid'] for e in entries)}/32 valid · {'best selected' if result['best'] else 'no valid grasp'} · preview only"
            )
            image = Image.new("RGB", (640, 300), COLORS.surface)
            draw = ImageDraw.Draw(image)
            lines = []
            for i, item in enumerate(entries[:4]):
                cx, cy = 80 + 160 * i, 145
                scale = 1700
                width, depth = self.size[:2] * scale
                draw.rectangle(
                    (cx - width / 2, cy - depth / 2, cx + width / 2, cy + depth / 2),
                    outline=COLORS.muted,
                    width=2,
                )
                w, x, y, z = item["quaternion_wxyz"]
                direction = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
                center = np.array([cx, cy]) + np.array(item["offset_m"][:2]) * scale
                half = direction * max(0, item["opening_m"]) * scale / 2
                color = "#63D7C4" if item["geometric_valid"] else COLORS.path
                draw.line(
                    (tuple(center - half), tuple(center + half)), fill=color, width=4
                )
                draw.text(
                    (cx - 55, 35),
                    "Best"
                    if i == 0 and result["best"] is not None
                    else f"Candidate {i + 1}",
                    fill=color,
                )
                draw.text(
                    (cx - 55, 230),
                    "Valid" if item["geometric_valid"] else "Rejected",
                    fill=color,
                )
                lines.append(
                    f"{i + 1}: center Z {item['offset_m'][2] * 1000:.1f} mm · opening {item['opening_m'] * 1000:.1f} mm"
                )
            self.plot.image = ImageTk.PhotoImage(image)
            self.plot.configure(image=self.plot.image)
            self.details.set(
                "\n".join(lines[:3])
                + (
                    "\nOutside training sizes · extrapolation, not validated"
                    if result["outside_training_range"]
                    else ""
                )
            )
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            self.status.set(str(exc))
        self.future = None

    def close(self):
        self.closed = True
        if self.pending is not None:
            self.window.after_cancel(self.pending)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()
