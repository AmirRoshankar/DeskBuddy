"""Offline depth fixture and a deliberately simple two-view arm drawing."""

import numpy as np
from PIL import Image, ImageDraw

from .geometry import deproject, transform_point

INTRINSICS = (600.0, 600.0, 320.0, 240.0)
T_BASE_CAMERA = np.array(
    [[1, 0, 0, 0.25], [0, -1, 0, 0], [0, 0, -1, 0.65], [0, 0, 0, 1.0]]
)


def synthetic_frame():
    image = Image.new("RGB", (640, 480), "#202023")
    d = ImageDraw.Draw(image)
    for x in range(0, 640, 40):
        d.line((x, 0, x, 480), fill="#35353A")
    for y in range(0, 480, 40):
        d.line((0, y, 640, y), fill="#35353A")
    for x, y in [(340, 240), (385, 195), (400, 275)]:
        d.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#55d9b0")
    d.text((16, 16), "SYNTHETIC DEPTH / TABLE PLANE", fill="white")
    d.text((16, 40), "Click a green target to plan a 6 cm hover", fill="#a3becd")
    return image, np.full((480, 640), 0.65)


def synthetic_point(u, v):
    return transform_point(T_BASE_CAMERA, deproject(u, v, 0.65, INTRINSICS))


def arm_image(arm, q, target=None, plan=None):
    image = Image.new("RGB", (500, 480), "#0e1820")
    d = ImageDraw.Draw(image)
    d.text((18, 14), "KINEMATIC PREVIEW / metres", fill="white")
    for label, vertical, top in [("SIDE: x / z", 2, 42), ("TOP: x / y", 1, 268)]:
        d.text((18, top), label, fill="#91b0c2")

        def project(p):
            return (
                25 + int(p[0] * 1000),
                top + 190 - int(p[vertical] * 600)
                if vertical == 2
                else top + 100 - int(p[1] * 500),
            )

        if vertical == 2:
            d.line((0, top + 190, 500, top + 190), fill="#586979")
        if plan is not None:
            path = [project(arm.fk(pose)) for pose in plan.joints_rad]
            d.line(path, fill="#43b6cf", width=2)
        points = [project(p) for p in arm.frames(q)[0]]
        d.line(points, fill="#ffbf69", width=5)
        for x, y in points:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#fff2d3")
        if target is not None:
            x, y = project(target)
            d.line((x - 7, y, x + 7, y), fill="#55d9b0", width=2)
            d.line((x, y - 7, x, y + 7), fill="#55d9b0", width=2)
    return image
