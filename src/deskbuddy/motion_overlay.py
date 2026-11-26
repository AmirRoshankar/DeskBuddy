"""Camera-projected model arm, planned motion, and selection-height outline."""

import numpy as np
from PIL import ImageDraw

PATH = "#F46D67"
WORKSPACE = "#63D7C4"
MODEL = "#F5F3F0"
JOINT = "#FF9275"
INK = "#171719"


def dashed(draw, a, b, color, width=1, dash=6):
    a, b = np.asarray(a), np.asarray(b)
    length = float(np.linalg.norm(b - a))
    if length < 1:
        return
    # Limit offscreen work as well as degenerate near-camera projections.
    if length > 10000:
        return
    for start in np.arange(0, length, dash * 2):
        p = a + (b - a) * start / length
        q = a + (b - a) * min(length, start + dash) / length
        draw.line((tuple(p), tuple(q)), fill=color, width=width)


def motion_geometry(arm, measured, plan):
    current = arm.frames(measured)[0] if measured is not None else None
    if plan is None:
        return current, None, None
    cached = getattr(plan, "_camera_motion", None)
    if cached is None or cached[0] is not arm:
        indices = np.unique(
            np.linspace(
                0, len(plan.joints_rad) - 1, min(80, len(plan.joints_rad))
            ).astype(int)
        )
        path = np.array([arm.fk(plan.joints_rad[i]) for i in indices])
        goal = arm.frames(plan.joints_rad[-1])[0]
        cached = (arm, path, goal)
        plan._camera_motion = cached
    return current, cached[1], cached[2]


def draw_motion(image, arm, measured, plan, config, projection):
    draw = ImageDraw.Draw(image)
    current, path, goal = motion_geometry(arm, measured, plan)

    def segments(points):
        projected = [projection(p) for p in points]
        for a, b in zip(projected[:-1], projected[1:]):
            if a is not None and b is not None:
                yield a, b

    if goal is not None:
        for a, b in segments(goal):
            dashed(draw, a, b, INK, 4)
            dashed(draw, a, b, PATH, 2)
    if path is not None:
        for a, b in segments(path):
            draw.line((a, b), fill=INK, width=6)
            draw.line((a, b), fill=PATH, width=3)
        # Direction arrows follow the actual projected trajectory, not a shortcut
        # between the current tip and target.
        for i in np.unique(
            np.linspace(1, len(path) - 1, min(3, len(path) - 1)).astype(int)
        ):
            a, b = projection(path[max(0, i - 3)]), projection(path[i])
            if a is None or b is None:
                continue
            vector = np.array(b) - a
            length = np.linalg.norm(vector)
            if length < 3:
                continue
            tangent = vector / length
            normal = np.array([-tangent[1], tangent[0]])
            tip = np.array(b)
            tail = tip - tangent * 8
            draw.polygon(
                [tuple(tip), tuple(tail + normal * 4), tuple(tail - normal * 4)],
                fill=PATH,
            )
    if current is not None:
        for a, b in segments(current):
            draw.line((a, b), fill=INK, width=6)
            draw.line((a, b), fill=MODEL, width=3)
        for point in current:
            pixel = projection(point)
            if pixel is None:
                continue
            x, y = pixel
            if 0 <= x < image.width and 0 <= y < image.height:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=JOINT, outline=INK)

    labels = []
    if current is not None:
        labels.insert(0, (MODEL, "MODEL ARM"))
    if plan is not None:
        labels.insert(1, (PATH, "PLAN / END POSE"))
    x, y = 10, image.height - 17
    for color, text in labels:
        draw.line((x, y + 5, x + 12, y + 5), fill=color, width=3)
        draw.text((x + 17, y), text, fill=color, stroke_width=1, stroke_fill=INK)
        x += 17 + int(draw.textlength(text)) + 18
