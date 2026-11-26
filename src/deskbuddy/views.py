"""Small, dependency-light depth and kinematic views for the desktop viewer."""

import numpy as np
from PIL import Image, ImageDraw

from .ui.theme import COLORS


def depth_images(depth):
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.clip((np.nan_to_num(depth, nan=0.0) - 0.2) / 0.8, 0, 1)
    palette = np.array(
        [[48, 18, 87], [45, 80, 167], [35, 174, 192], [155, 217, 95], [253, 231, 37]],
        float,
    )
    index = scaled * 4
    low = np.minimum(index.astype(int), 3)
    fraction = (index - low)[..., None]
    rgb = palette[low] * (1 - fraction) + palette[low + 1] * fraction
    rgb[~valid] = 0
    gray = ((1 - scaled) * 255).astype(np.uint8)
    gray[~valid] = 0
    return Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(gray)


def arm_panel(
    arm, q, target=None, plan=None, width=560, height=240, table_z=0.0, paths=None
):
    image = Image.new("RGB", (width, height), COLORS.surface)
    draw = ImageDraw.Draw(image)
    pane = width // 2
    for label, vertical, column in (("SIDE  x / z", 2, 0), ("TOP  x / y", 1, 1)):
        ox = column * pane
        draw.text((ox + 12, 10), label, fill=COLORS.muted)

        def project(p):
            x = ox + 24 + (p[0] + 0.1) * (pane - 44) / 0.6
            value = (p[2] + 0.08) / 0.55 if vertical == 2 else (p[1] + 0.3) / 0.6
            y = height - 25 - value * (height - 60)
            return float(x), float(y)

        if vertical == 2:
            y = project([0, 0, table_z])[1]
            draw.line((ox + 8, y, ox + pane - 8, y), fill=COLORS.border)
        if plan is not None:
            step = max(1, len(plan.joints_rad) // 40)
            points = [project(arm.fk(p)) for p in plan.joints_rad[::step]]
            if len(points) > 1:
                draw.line(points, fill=COLORS.path, width=2)
        from .motion_overlay import dashed

        for entry in paths or []:
            if entry["points_m"] is None:
                continue
            route = [project(p) for p in entry["points_m"]]
            for a, b in zip(route[:-1], route[1:]):
                if entry["valid"]:
                    draw.line((a, b), fill=entry["color"], width=2)
                else:
                    dashed(draw, a, b, entry["color"], width=1, dash=4)
        points = [project(p) for p in arm.frames(q)[0]]
        draw.line(points, fill=COLORS.text, width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS.accent_light)
        if target is not None:
            x, y = project(target)
            draw.line((x - 6, y, x + 6, y), fill="#66e2ad", width=2)
            draw.line((x, y - 6, x, y + 6), fill="#66e2ad", width=2)
    if paths:
        from .trajectory.path_overlays import COLORS as PATH_COLORS

        x = 12
        for name, color in PATH_COLORS.items():
            draw.line((x, height - 12, x + 18, height - 12), fill=color, width=3)
            draw.text(
                (x + 23, height - 19),
                "iMF" if name == "imf" else name.upper(),
                fill=color,
            )
            x += 94
        draw.text((x, height - 19), "Go reference", fill=COLORS.path)
    return image
