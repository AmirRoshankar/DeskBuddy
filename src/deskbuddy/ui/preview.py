"""Export a synthetic dashboard preview from our own widget tree.

This renders application-owned fixture data with Pillow; it does not capture
the desktop, inspect another process, open a camera, or connect to a robot.
Native text rasterization can differ slightly from this portable export.
"""

import argparse
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from ..config import load_config
from ..demo import T_BASE_CAMERA, synthetic_frame, synthetic_point
from ..kinematics import Arm
from ..planning import plan_reach
from ..trajectory.path_overlays import COLORS as PATH_COLORS
from ..trajectory.perception import DetectedBox
from ..views import depth_images
from .layout import build
from .scene_inspector import OrbitView, render_scene
from .theme import COLORS
from .viewport import CameraViewport


def _font(root, widget):
    actual = tkfont.Font(root=root, font=widget.cget("font")).actual()
    size = actual["size"]
    size = int(abs(size) if size < 0 else size * root.winfo_fpixels("1i") / 72)
    mono = "consolas" in actual["family"].lower()
    name = (
        "consola.ttf"
        if mono
        else ("segoeuib.ttf" if actual["weight"] == "bold" else "segoeui.ttf")
    )
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except OSError:
        return ImageFont.load_default(size=size)


def _lines(text, font, width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = (line + " " + word).strip()
            if line and font.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def render(root):
    """Use widget geometry and contents; never read screen pixels."""
    image = Image.new("RGB", (root.winfo_width(), root.winfo_height()), COLORS.canvas)

    def visit(widget, x=0, y=0):
        if isinstance(widget, tk.Menu):
            return
        if widget is not root:
            x += widget.winfo_x()
            y += widget.winfo_y()
        w, h = widget.winfo_width(), widget.winfo_height()
        draw = ImageDraw.Draw(image)
        try:
            draw.rectangle((x, y, x + w - 1, y + h - 1), fill=widget.cget("bg"))
            if int(widget.cget("highlightthickness")):
                draw.rectangle(
                    (x, y, x + w - 1, y + h - 1),
                    outline=widget.cget("highlightbackground"),
                )
        except tk.TclError:
            pass
        if isinstance(widget, tk.Canvas):
            for item in widget.find_all():
                kind = widget.type(item)
                coordinates = widget.coords(item)
                coordinates = [
                    value + (x if i % 2 == 0 else y)
                    for i, value in enumerate(coordinates)
                ]
                color = widget.itemcget(item, "fill")
                if kind == "line":
                    draw.line(
                        coordinates,
                        fill=color,
                        width=max(1, round(float(widget.itemcget(item, "width")))),
                    )
                elif kind == "oval":
                    draw.ellipse(coordinates, fill=color)
        if isinstance(widget, tk.Radiobutton):
            if str(root.getvar(widget.cget("variable"))) == str(widget.cget("value")):
                draw.rectangle(
                    (x + 1, y + 1, x + w - 2, y + h - 2),
                    fill=widget.cget("selectcolor"),
                )
        if isinstance(
            widget,
            (
                tk.Label,
                tk.Button,
                tk.Menubutton,
                tk.Radiobutton,
                tk.Checkbutton,
                tk.Entry,
                tk.Text,
            ),
        ):
            if getattr(widget, "image", None) is not None:
                pixels = ImageTk.getimage(widget.image).convert("RGB")
                image.paste(
                    pixels, (x + (w - pixels.width) // 2, y + (h - pixels.height) // 2)
                )
            else:
                text = (
                    widget.get("1.0", "end-1c")
                    if isinstance(widget, tk.Text)
                    else (
                        widget.get()
                        if isinstance(widget, tk.Entry)
                        else widget.cget("text")
                    )
                )
                if not isinstance(widget, tk.Text) and widget.cget("textvariable"):
                    text = root.getvar(widget.cget("textvariable"))
                if text:
                    font = _font(root, widget)
                    try:
                        wrap = int(widget.cget("wraplength"))
                    except tk.TclError:
                        wrap = w - 8
                    lines = (
                        _lines(text, font, min(w - 8, wrap or w - 8))
                        if wrap or isinstance(widget, tk.Text)
                        else text.split("\n")
                    )
                    line_height = font.getbbox("Ag")[3] + 4
                    anchor = (
                        "nw"
                        if isinstance(widget, tk.Text)
                        else (
                            "w"
                            if isinstance(widget, tk.Entry)
                            else widget.cget("anchor")
                        )
                    )
                    oy = (
                        3
                        if anchor in ("n", "ne", "nw")
                        else max(0, (h - line_height * len(lines)) // 2)
                    )
                    layer = Image.new("RGBA", (w, h))
                    painter = ImageDraw.Draw(layer)
                    for line in lines:
                        ox = (
                            4
                            if anchor in ("w", "nw", "sw")
                            else (w - font.getlength(line)) / 2
                        )
                        if anchor in ("e", "ne", "se"):
                            ox = w - 4 - font.getlength(line)
                        painter.text((ox, oy), line, font=font, fill=widget.cget("fg"))
                        oy += line_height
                    image.paste(layer, (x, y), layer)
        for child in widget.winfo_children():
            visit(child, x, y)

    visit(root)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/assets/dashboard-preview.png")
    parser.add_argument("--can-task", action="store_true")
    parser.add_argument("--size", help="Optional WIDTHxHEIGHT fixture window size")
    parser.add_argument(
        "--comparison", action="store_true", help="Export the comparison window fixture"
    )
    parser.add_argument(
        "--bounds", action="store_true", help="Export the bounds editor fixture"
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Show the calibration guidance fixture",
    )
    args = parser.parse_args()
    root = tk.Tk()
    root.attributes("-alpha", 0)
    if args.comparison:
        from .comparison import ComparisonWindow
        from .theme import apply

        apply(root)
        config = load_config()
        config.update(T_base_camera=T_BASE_CAMERA.tolist(), calibration_verified=True)
        rgb, depth = synthetic_frame()
        intrinsics = dict(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        q = np.deg2rad(config["preview_start_deg"])
        box = DetectedBox(
            np.array([0.20, -0.06, 0]),
            np.array([0.31, 0.06, 0.1]),
            1000,
            0.1,
            (200, 150, 400, 300),
        )

        def provider():
            return np.asarray(rgb), depth, intrinsics, config, q, [box]

        provider.arm = Arm()
        view = ComparisonWindow(root, provider, lambda: None)
        view.window.attributes("-alpha", 0)
        view.message.set("Synthetic UI fixture · no hardware or model execution")
        if args.size:
            view.window.geometry(args.size)
        root.update()
        view.refresh()
        root.update_idletasks()
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        render(view.window).save(path)
        view.close()
        root.destroy()
        return
    if args.bounds:
        from .theme import apply
        from .workspace import open_editor

        apply(root)
        window = open_editor(root, load_config(), lambda _: None)
        window.attributes("-alpha", 0)
        root.update()
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        render(window).save(path)
        root.destroy()
        return
    view = build(
        root,
        initial_status="Offline design preview · synthetic scene · no hardware connected.",
        controls=True,
        calibration_mode=False,
        on_go=lambda: None,
        on_stop=lambda: None,
        on_recalibrate=lambda: None,
        hover_m=load_config()["hover_m"],
        task_mode="can_discard" if args.can_task else "reach",
    )
    if args.size:
        root.geometry(args.size)
    view.torque_status.set("Offline fixture · motor commands disabled")
    view.telemetry_status.set("250 Hz requested · encoder feedback controller")
    view.fault_status.set("Synthetic data · fixed scene estimate · no live telemetry")
    view.cursor_info.set("Pixel 385, 195\nSurface estimate\nCached scene depth")
    view.go.configure(state="normal")
    if args.calibration:
        from .calibration import CalibrationBar

        view.go.configure(state="disabled")
        CalibrationBar(root).step(
            "1 / 4 · Clamp the base and support the arm so it cannot fall. Release hold to reposition it by hand. No reach will run.",
            "Arm supported — release hold",
        )
    rgb, depth = synthetic_frame()
    # Illustrative depth variation so the exported palette is legible. This
    # fixture is labelled synthetic and is never used for geometric validation.
    depth = np.broadcast_to(np.linspace(0.42, 0.86, 480)[:, None], (480, 640)).copy()
    depth[170:250, 290:370] = 0.39
    config = load_config()
    config["T_base_camera"] = T_BASE_CAMERA.tolist()
    if args.can_task:
        config["discard_zone"] = [
            [0.16, -0.15, 0],
            [0.27, -0.15, 0],
            [0.27, -0.05, 0],
            [0.16, -0.05, 0],
        ]
    q = np.deg2rad(config["preview_start_deg"])
    arm = Arm()
    target = synthetic_point(385, 195) + [0, 0, config["hover_m"]]
    plan = plan_reach(arm, q, target, config)
    color_depth = depth_images(depth)[0]
    baseline = np.array(
        [
            arm.fk(pose)
            for pose in plan.joints_rad[:: max(1, len(plan.joints_rad) // 40)]
        ]
    )
    paths = []
    for i, (method, color) in enumerate(PATH_COLORS.items()):
        points = baseline.copy()
        phase = np.sin(np.linspace(0, np.pi, len(points)))
        points[:, 1] += (i - 1.5) * 0.018 * phase
        points[:, 2] += i * 0.008 * phase
        paths.append(
            dict(
                method=method,
                color=color,
                valid=True,
                points_m=points.tolist(),
                reason=None,
            )
        )

    root.update_idletasks()
    scene = dict(
        arm=arm,
        q=q,
        config=config,
        plan=plan,
        paths=paths,
        selected_method="imf",
        pose_source="synthetic fixture · no live encoders",
        boxes=[]
        if not args.can_task
        else [
            DetectedBox(
                np.array([0.27, 0.04, 0]),
                np.array([0.336, 0.106, 0.122]),
                200,
                0.122,
                (100, 100, 200, 250),
            )
        ],
    )

    def show(label, pixels):
        if label in (view.zoom, view.depth_zoom):
            label.zoom_source = pixels
            side = getattr(label, "zoom_side", 144)
            pixels = pixels.resize((side, side))
        if label in (view.color, view.depth, view.webcam):
            viewport = CameraViewport(
                max(1, label.winfo_width()),
                max(1, label.winfo_height()),
                pixels.width,
                pixels.height,
            )
            pixels = viewport.render(pixels)
        label.image = ImageTk.PhotoImage(pixels)
        label.configure(image=label.image)

    show(view.color, rgb)
    show(view.depth, color_depth)
    show(view.webcam, rgb.copy())
    show(
        view.diagram,
        render_scene(
            scene,
            OrbitView(),
            (
                max(160, view.diagram.winfo_width()),
                max(120, view.diagram.winfo_height()),
            ),
            envelopes=False,
        ),
    )
    show(view.zoom, rgb.crop((315, 190, 363, 238)).resize((144, 144)))
    show(view.depth_zoom, color_depth.crop((315, 190, 363, 238)).resize((144, 144)))
    root.update()
    show(view.color, rgb)
    show(view.depth, color_depth)
    show(view.webcam, rgb.copy())
    show(
        view.diagram,
        render_scene(
            scene,
            OrbitView(),
            (view.diagram.winfo_width(), view.diagram.winfo_height()),
            envelopes=False,
        ),
    )
    root.update_idletasks()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    render(root).save(path)
    print(
        f"Exported app-owned fixture preview: {path} ({root.winfo_width()} × {root.winfo_height()})"
    )
    root.destroy()


if __name__ == "__main__":
    main()
