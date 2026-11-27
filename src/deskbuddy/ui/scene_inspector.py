"""Read-only native 3D inspection of the exact planning geometry."""

import itertools
import time
import tkinter as tk
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ..test_bed import corners
from .theme import COLORS, FONT_SMALL
from .widgets import ControlButton


class RenderGeometry:
    """View-independent display data. Planning/control collision checks stay exact."""

    def __init__(self):
        self.arm = None
        self.poses = OrderedDict()
        self.plan = None
        self.route = None

    def pose(self, arm, q):
        if self.arm is not arm:
            self.arm = arm
            self.poses.clear()
            self.plan = None
        key = tuple(np.asarray(q, float))
        if key not in self.poses:
            from ..self_collision import collisions, world_parts

            self.poses[key] = (
                arm.frames(q)[0],
                world_parts(arm, q)[0],
                collisions(arm, q),
            )
            if len(self.poses) > 4:
                self.poses.popitem(last=False)
        self.poses.move_to_end(key)
        return self.poses[key]

    def path(self, arm, plan):
        # UI plans are immutable snapshots, replaced when the target changes.
        if self.plan is not plan:
            indices = np.unique(
                np.linspace(
                    0,
                    len(plan.joints_rad) - 1,
                    min(300, len(plan.joints_rad)),
                    dtype=int,
                )
            )
            self.route = np.array([arm.fk(plan.joints_rad[i]) for i in indices])
            self.plan = plan
        return self.route


class OrbitView:
    def __init__(self):
        self.yaw, self.pitch = 0.7, 0.65
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.geometry = RenderGeometry()
        self.framing = None
        self.bounds_signature = None
        # Held while a bounds handle is dragged so the view does not re-frame
        # (and move the handle away from the cursor) on every edit.
        self.freeze_framing = False

    def rotate(self, dx, dy):
        self.yaw += dx * 0.008
        self.pitch = float(np.clip(self.pitch + dy * 0.008, 0.1, np.pi / 2 - 0.02))

    def scroll(self, delta):
        self.zoom = float(np.clip(self.zoom * np.exp(delta / 120 * 0.12), 0.2, 8))

    def project(self, points, center, scale, size):
        cy, sy, cp, sp = (
            np.cos(self.yaw),
            np.sin(self.yaw),
            np.cos(self.pitch),
            np.sin(self.pitch),
        )
        right, up = np.array([cy, -sy, 0]), np.array([sp * sy, sp * cy, cp])
        relative = np.asarray(points) - center
        return (
            np.c_[relative @ right, -(relative @ up)] * scale * self.zoom
            + np.array(size) / 2
            + self.pan
        )


def scene_projection(snapshot, size, view=None):
    low, high = (
        np.array(snapshot["config"]["workspace_min_m"]),
        np.array(snapshot["config"]["workspace_max_m"]),
    )
    signature = (tuple(low), tuple(high))
    if (
        view is not None
        and view.framing is not None
        and (view.bounds_signature == signature or view.freeze_framing)
    ):
        center, extent = view.framing
        return center, min(size) * 0.8 / extent
    points = snapshot["arm"].frames(snapshot["q"])[0]
    visible = np.vstack(
        [points, np.vstack([low, high])]
        + [np.vstack([b.low, b.high]) for b in snapshot.get("boxes", [])]
    )
    center = (visible.min(0) + visible.max(0)) / 2
    extent = max(float(np.linalg.norm(visible.max(0) - visible.min(0))), 0.3)
    if view is not None:
        view.framing = (center.copy(), extent)
        view.bounds_signature = signature
    return center, min(size) * 0.8 / extent


def bounds_handle_at(snapshot, view, size, pixel, radius=12):
    """The workspace handle drawn nearest ``pixel`` in the 3D view, or None."""
    from ..workspace_geometry import box_handles

    center, scale = scene_projection(snapshot, size, view)
    best = None
    for handle in box_handles(snapshot["config"]):
        x, y = view.project(np.atleast_2d(handle[1]), center, scale, size)[0]
        distance = float(np.hypot(x - pixel[0], y - pixel[1]))
        if distance <= radius and (best is None or distance < best[0]):
            best = (distance, handle)
    return None if best is None else best[1]


def drag_bounds(snapshot, view, size, handle, press, pixel, original):
    """New workspace bounds for a handle dragged from ``press`` to ``pixel``.

    Face handles move along their axis by the mouse motion projected onto that
    axis on screen; the centre handle moves the box in XY on its own height.
    Absolute from the press position, so nothing accumulates.
    """
    _, anchor, axis, side = handle
    center, scale = scene_projection(snapshot, size, view)
    low = np.array(original["workspace_min_m"], float)
    high = np.array(original["workspace_max_m"], float)
    motion = np.asarray(pixel, float) - np.asarray(press, float)
    if axis is None:
        a = pick_on_plane(snapshot, view, size, press, anchor[2])
        b = pick_on_plane(snapshot, view, size, pixel, anchor[2])
        delta = b - a
        delta[2] = 0
        return (low + delta).tolist(), (high + delta).tolist()
    step = np.zeros(3)
    step[axis] = 0.1
    screen = (
        view.project(np.vstack([anchor, anchor + step]), center, scale, size)[1]
        - view.project(np.atleast_2d(anchor), center, scale, size)[0]
    )
    if screen @ screen < 4:
        raise ValueError("Rotate the view: this axis points at the screen")
    delta = float(motion @ screen / (screen @ screen) * 0.1)
    if side == 0:
        low[axis] = min(low[axis] + delta, high[axis] - 0.01)
    else:
        high[axis] = max(high[axis] + delta, low[axis] + 0.01)
    return low.tolist(), high.tolist()


def pick_on_plane(snapshot, view, size, pixel, z):
    center, scale = scene_projection(snapshot, size, view)
    cy, sy, cp, sp = (
        np.cos(view.yaw),
        np.sin(view.yaw),
        np.cos(view.pitch),
        np.sin(view.pitch),
    )
    right, up = np.array([cy, -sy, 0]), np.array([sp * sy, sp * cy, cp])
    normal = np.cross(right, up)
    if abs(normal[2]) < 0.08:
        raise ValueError(
            "Rotate the 3D view slightly above the table to select targets"
        )
    xy = (np.asarray(pixel) - np.array(size) / 2 - view.pan) / (scale * view.zoom)
    origin = center + right * xy[0] - up * xy[1]
    return origin + normal * ((z - origin[2]) / normal[2])


def pick_in_workspace(snapshot, view, size, pixel, preferred_z):
    """Clip the displayed pixel ray to the actual box before selecting height.

    A pixel inside a projected volume need not hit the preferred horizontal slice.
    Keep the pixel ray and choose its nearest valid height, rather than inventing
    an out-of-bounds XY point on that slice.
    """
    center, scale = scene_projection(snapshot, size, view)
    cy, sy, cp, sp = (
        np.cos(view.yaw),
        np.sin(view.yaw),
        np.cos(view.pitch),
        np.sin(view.pitch),
    )
    right, up = np.array([cy, -sy, 0.0]), np.array([sp * sy, sp * cy, cp])
    direction = np.cross(right, up)
    screen = (np.asarray(pixel) - np.array(size) / 2 - view.pan) / (scale * view.zoom)
    origin = center + right * screen[0] - up * screen[1]
    lo = np.array(snapshot["config"]["workspace_min_m"], float)
    hi = np.array(snapshot["config"]["workspace_max_m"], float)
    enter, leave = -np.inf, np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if origin[i] < lo[i] - 1e-9 or origin[i] > hi[i] + 1e-9:
                raise ValueError("Click inside the projected workspace box")
        else:
            a, b = sorted(
                ((lo[i] - origin[i]) / direction[i], (hi[i] - origin[i]) / direction[i])
            )
            enter, leave = max(enter, a), min(leave, b)
    if enter > leave + 1e-9:
        raise ValueError("Click inside the projected workspace box")
    preferred = (
        (preferred_z - origin[2]) / direction[2]
        if abs(direction[2]) > 1e-12
        else (enter + leave) / 2
    )
    return np.clip(origin + direction * np.clip(preferred, enter, leave), lo, hi)


def render_scene(snapshot, view, size=(900, 650), envelopes=True, phase=1.0):
    """App-owned image rendering, also usable for deterministic visual fixtures."""
    arm, q, config = snapshot["arm"], snapshot["q"], snapshot["config"]
    low, high = np.array(config["workspace_min_m"]), np.array(config["workspace_max_m"])
    points = arm.frames(q)[0]
    center, scale = scene_projection(snapshot, size, view)
    image = Image.new("RGB", size, COLORS.canvas)
    draw = ImageDraw.Draw(image, "RGBA")

    def project(p):
        return [tuple(x) for x in view.project(np.atleast_2d(p), center, scale, size)]

    def line(p, color, width=1):
        draw.line(project(p), fill=color, width=width)

    def box(lo, hi, color, fill=None):
        vertices = np.array(list(itertools.product(*zip(lo, hi))))
        if fill:
            for axis in range(3):
                for side in (lo[axis], hi[axis]):
                    face = vertices[np.isclose(vertices[:, axis], side)]
                    xy = np.array(project(face))
                    angles = np.arctan2(
                        xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean()
                    )
                    draw.polygon([tuple(p) for p in xy[np.argsort(angles)]], fill=fill)
        for a, b in itertools.combinations(vertices, 2):
            if np.count_nonzero(a != b) == 1:
                line([a, b], color)

    # Table is the same tilted plane used by the planner, extended over bounds.
    xy = np.array(
        [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
    )
    import cv2

    from ..table_collision import physical_plane

    normal, offset = physical_plane(config)
    table = np.c_[xy, -(xy @ normal[:2] + offset) / normal[2]]
    draw.polygon(project(table), fill=(87, 191, 169, 30), outline="#577b70")
    for t in np.linspace(0, 1, 7):
        line(
            [table[0] * (1 - t) + table[1] * t, table[3] * (1 - t) + table[2] * t],
            "#344c45",
        )
        line(
            [table[0] * (1 - t) + table[3] * t, table[1] * (1 - t) + table[2] * t],
            "#344c45",
        )
    paper = corners(config, inset=False)
    if paper is not None:
        draw.polygon(project(paper), fill=(235, 232, 218, 38), outline="#e6ddbd")
    box(low, high, "#527d91")
    if config.get("show_workspace", True):
        from ..workspace_geometry import box_handles

        for name, point, axis, _ in box_handles(config):
            x, y = project(point)[0]
            half = 5 if axis is None else 4
            draw.rectangle(
                (x - half, y - half, x + half, y + half),
                fill="#16222a",
                outline="#77bceb",
                width=2,
            )
    for index, obstacle in enumerate(snapshot.get("boxes", []), 1):
        lo, hi = obstacle.low, obstacle.high
        box(lo, hi, "#ffb875", (239, 143, 76, 35))
        x, y = project(hi)[0]
        dimensions = (hi - lo) * 100
        draw.text(
            (x + 5, y - 16),
            f"{index}: {dimensions[0]:.1f} × {dimensions[1]:.1f} × {dimensions[2]:.1f} cm",
            fill="#ffcc9b",
        )

    zone = config.get("discard_zone")
    if zone is not None:
        draw.polygon(project(zone), fill=(193, 154, 255, 30), outline="#C19AFF")
        x, y = project(np.asarray(zone).mean(0))[0]
        draw.text((x + 5, y), "Discard zone", fill="#C19AFF")
    selected = snapshot.get("discard_points") or []
    if selected:
        if len(selected) > 1:
            line(selected, "#C19AFF", 2)
        for i, (x, y) in enumerate(project(selected), 1):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#C19AFF")
            draw.text((x + 7, y), str(i), fill="#C19AFF")

    def capsules(points):
        angles = np.linspace(0, 2 * np.pi, 17)
        for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
            radius = 0.045 if i == len(points) - 2 else 0.035
            direction = b - a
            direction = direction / max(np.linalg.norm(direction), 1e-12)
            side = np.cross(direction, [0, 0, 1])
            if np.linalg.norm(side) < 0.001:
                side = np.array([1.0, 0, 0])
            side /= np.linalg.norm(side)
            other = np.cross(direction, side)
            ring = radius * (
                np.cos(angles[:, None]) * side + np.sin(angles[:, None]) * other
            )
            line(a + ring, "#55565f")
            line(b + ring, "#55565f")
            for offset in ring[::4]:
                line([a + offset, b + offset], "#55565f")
            for endpoint in (a, b):
                for axis in range(3):
                    sphere = np.zeros((len(angles), 3))
                    sphere[:, axis] = np.cos(angles) * radius
                    sphere[:, (axis + 1) % 3] = np.sin(angles) * radius
                    line(endpoint + sphere, "#45464e")

    _, parts, hits = view.geometry.pose(arm, q)

    def body_outlines(pose, color):
        for part in view.geometry.pose(arm, pose)[1]:
            xy = view.project(part["vertices"], center, scale, size)
            hull = cv2.convexHull(np.asarray(xy, np.float32))[:, 0]
            draw.polygon([tuple(p) for p in hull], fill=color + "18", outline=color)

    body_outlines(q, "#95989e")
    if hits:
        names = {hit[key] for hit in hits for key in ("first", "second")}
        for part in parts:
            if part["name"] in names:
                xy = view.project(part["vertices"], center, scale, size)
                hull = cv2.convexHull(np.asarray(xy, np.float32))[:, 0]
                draw.polygon(
                    [tuple(p) for p in hull],
                    fill=(255, 83, 75, 45),
                    outline="#ff534b",
                )
    if envelopes:
        capsules(points)
    line(points, "#f5f3f0", 5)
    for x, y in project(points):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ff9275")
    plan = snapshot.get("plan")
    if plan is not None:
        route = view.geometry.path(arm, plan)
        line(route, "#f46d67", 3)
        transfer = next(
            (i["can_transfer"] for i in plan.planner_info or [] if "can_transfer" in i),
            None,
        )
        if transfer:
            can = transfer["can"]
            base = np.array(can["center_m"])
            radius = can["diameter_m"] / 2
            angles = np.linspace(0, 2 * np.pi, 33)
            ring = np.c_[
                np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(len(angles))
            ]
            line(base + ring, "#77cbe8", 2)
            line(base + ring + [0, 0, can["height_m"]], "#77cbe8", 2)
            for offset in ring[::8]:
                line(
                    [base + offset, base + offset + [0, 0, can["height_m"]]], "#77cbe8"
                )
            # The real lip: the narrower 202 lid the rim/hole grasps target.
            from ..can_discard import lid_radius

            lid = ring * (lid_radius(can) / radius)
            line(base + lid + [0, 0, can["height_m"]], "#d8dde3", 2)
        grasp = next(
            (i for i in plan.planner_info or [] if "grasp_pose_rad" in i), None
        )
        if grasp:
            contact = np.array(grasp["contact_m"])
            line([grasp["pregrasp_m"], contact], "#77cbe8", 3)
            line(arm.frames(grasp["grasp_pose_rad"])[0], "#77cbe8", 2)
            line([grasp["wrist_m"], contact], "#77cbe8", 5)
            body_outlines(
                grasp["grasp_pose_rad"],
                "#ff7369" if grasp.get("table_clearance_m", 0) < 0 else "#77cbe8",
            )
            x, y = project(contact)[0]
            draw.text(
                (x + 5, y),
                f"Grasp tilt {grasp['tilt_deg']:.1f}° · table {grasp.get('table_clearance_m', 0) * 1000:.1f} mm",
                fill="#77cbe8",
            )
        index = int(np.clip(phase, 0, 1) * (len(plan.joints_rad) - 1))
        ghost = arm.frames(plan.joints_rad[index])[0]
        if envelopes:
            capsules(ghost)
        line(ghost, "#f46d67", 2)
        for info in plan.planner_info or []:
            for collision in info.get("collision_predictions", []):
                x, y = project(collision["tip_m"])[0]
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#ff534b", width=3)
    from ..trajectory.path_overlays import draw_paths

    draw_paths(
        image,
        snapshot.get("paths", []),
        lambda p: project(p)[0],
        snapshot.get("selected_method"),
    )
    target = snapshot.get("target")
    if target is not None:
        x, y = project(target)[0]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#63D7C4", width=2)
        draw.text(
            (x + 10, y - 12), snapshot.get("target_label", "Target"), fill="#63D7C4"
        )
    objective = snapshot.get("objective")
    if objective is not None:
        hole = np.asarray(objective["point_m"], float)
        angles = np.linspace(0, 2 * np.pi, 25)
        rim = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))] * 0.007
        line(hole + rim, "#ffd166", 3)
        for axis in (np.array([0.012, 0, 0]), np.array([0, 0.012, 0])):
            line([hole - axis, hole + axis], "#ffd166", 2)
        if objective.get("depth_m", 0) > 0:
            # Where the in-hole fingertip is planned to stop.
            bottom = hole - [0, 0, objective["depth_m"]]
            line([hole + [0, 0, 0.03], bottom], "#ffd166", 2)
            x, y = project(bottom)[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#ffd166")
        x, y = project(hole)[0]
        draw.text((x + 12, y - 16), objective["label"], fill="#ffd166")
    origin = np.zeros(3)
    for axis, color, label in zip(
        np.eye(3) * 0.08, ["#fa8276", "#72caa6", "#77bceb"], ["X", "Y", "Z"]
    ):
        line([origin, axis], color, 2)
        x, y = project(axis)[0]
        draw.text((x + 4, y), label, fill=color)
    draw.text(
        (14, 14),
        "Base frame · metres · shared planning geometry",
        fill="#f5f3f0",
    )
    draw.text(
        (14, 32),
        "Orange: objects · blue: bounds · cream: A4 · grey: robot bodies",
        fill="#b1afad",
    )
    draw.text(
        (14, size[1] - 46),
        "Arm: " + snapshot.get("pose_source", "model estimate"),
        fill="#b1afad",
    )
    if hits:
        draw.text(
            (14, 52),
            "Predicted self overlap: " + ", ".join(sorted(names)),
            fill="#ff7369",
        )
    return image


class SceneNavigation:
    """Modifier-gated navigation; all handlers are presentation-only."""

    def __init__(
        self,
        label,
        changed=lambda: None,
        require_ctrl=True,
        on_pick=lambda event: None,
        on_hover=lambda event: None,
        on_handle=None,
    ):
        self.on_pick, self.on_hover = on_pick, on_hover
        # on_handle(phase, event) with phase "press"/"drag"/"release"; a true
        # result from "press" captures the drag for a bounds handle.
        self.on_handle, self.handle_drag = on_handle, False
        self.pick_start = None
        self.label, self.changed, self.require_ctrl = label, changed, require_ctrl
        self.view = OrbitView()
        self.pending_redraw = None
        self.last_redraw = 0.0
        self.drag = None
        for button in (1, 3):
            label.bind(
                f"<ButtonPress-{button}>", lambda event, b=button: self.begin(event, b)
            )
            label.bind(
                f"<B{button}-Motion>", lambda event, b=button: self.move(event, b)
            )
            label.bind(f"<ButtonRelease-{button}>", self.release)
        label.bind("<MouseWheel>", self.wheel)
        label.bind("<Control-Double-Button-1>", self.reset)

    def request_redraw(self):
        if not hasattr(self.label, "after"):
            self.changed()
            return
        if self.pending_redraw is None:
            delay = max(1, int((self.last_redraw + 0.016 - time.monotonic()) * 1000))
            self.pending_redraw = self.label.after(delay, self.flush)

    def flush(self):
        self.last_redraw = time.monotonic()
        self.pending_redraw = None
        if not hasattr(self.label, "winfo_exists") or self.label.winfo_exists():
            self.changed()

    def enabled(self, event):
        return not self.require_ctrl or bool(event.state & 0x4)

    def begin(self, event, button):
        if (
            button == 1
            and not self.enabled(event)
            and self.on_handle is not None
            and self.on_handle("press", event)
        ):
            self.handle_drag, self.drag, self.pick_start = True, None, None
            return "break"
        self.drag = (event.x, event.y) if self.enabled(event) else None
        self.pick_start = (
            (event.x, event.y) if button == 1 and not self.enabled(event) else None
        )
        return "break"

    def move(self, event, button):
        if self.handle_drag:
            self.on_handle("drag", event)
            return "break"
        if (
            self.pick_start is not None
            and np.linalg.norm(np.array([event.x, event.y]) - self.pick_start) > 4
        ):
            self.pick_start = None
        if self.drag is not None and self.enabled(event):
            dx, dy = event.x - self.drag[0], event.y - self.drag[1]
            if button == 1:
                self.view.rotate(dx, dy)
            else:
                self.view.pan += [dx, dy]
            self.drag = (event.x, event.y)
            self.request_redraw()
        elif not self.enabled(event):
            self.drag = None
        return "break"

    def release(self, event):
        if self.handle_drag:
            self.handle_drag = False
            self.on_handle("release", event)
            return "break"
        if self.pick_start is not None and not self.enabled(event):
            self.on_pick(event)
        self.pick_start = None
        self.drag = None
        return "break"

    def wheel(self, event):
        if self.enabled(event):
            self.view.scroll(event.delta)
            self.request_redraw()
        else:
            self.on_hover(event)
        return "break"

    def reset(self, event=None):
        self.view = OrbitView()
        self.request_redraw()
        return "break"


class SceneInspector:
    def __init__(self, root, provider):
        self.provider = provider
        self.view = OrbitView()
        self.last_render_key = None
        self.window = tk.Toplevel(root)
        self.window.title("3D scene · estimates")
        self.window.geometry("1000x760")
        self.window.minsize(650, 480)
        bar = tk.Frame(self.window)
        bar.pack(fill="x", padx=12, pady=8)
        ControlButton(bar, text="Reset view", command=self.reset).pack(side="left")
        self.envelopes = tk.BooleanVar(value=True)
        tk.Checkbutton(
            bar,
            text="Clearance envelopes",
            variable=self.envelopes,
            selectcolor=COLORS.raised,
            highlightthickness=0,
        ).pack(side="left", padx=12)
        tk.Label(
            bar,
            text="Ctrl + left drag: rotate · Ctrl + right drag: pan · Ctrl + wheel: zoom",
            font=FONT_SMALL,
        ).pack(side="right")
        self.image = tk.Label(self.window, bg=COLORS.canvas, width=1, height=1)
        self.image.pack(fill="both", expand=True)
        self.phase = tk.DoubleVar(value=1.0)
        tk.Scale(
            self.window,
            from_=0,
            to=1,
            resolution=0.001,
            orient="horizontal",
            variable=self.phase,
            label="Planned pose",
            showvalue=False,
            highlightthickness=0,
            bg=COLORS.canvas,
            fg=COLORS.text,
            troughcolor=COLORS.raised,
        ).pack(fill="x", padx=12, pady=8)
        self.drag = None
        for button in (1, 3):
            self.image.bind(f"<ButtonPress-{button}>", self.begin)
            self.image.bind(
                f"<B{button}-Motion>", lambda event, b=button: self.move(event, b)
            )
        self.image.bind(
            "<MouseWheel>",
            lambda event: (
                self.view.scroll(event.delta) if event.state & 0x4 else "break"
            ),
        )
        self.refresh()

    def reset(self):
        self.view = OrbitView()
        self.last_render_key = None

    def begin(self, event):
        self.drag = (event.x, event.y) if event.state & 0x4 else None

    def move(self, event, button):
        if not event.state & 0x4:
            self.drag = None
            return
        if self.drag is not None:
            dx, dy = event.x - self.drag[0], event.y - self.drag[1]
            if button == 1:
                self.view.rotate(dx, dy)
            else:
                self.view.pan += [dx, dy]
        self.drag = (event.x, event.y)

    def refresh(self):
        if not self.window.winfo_exists():
            return
        size = (max(100, self.image.winfo_width()), max(100, self.image.winfo_height()))
        snapshot = self.provider()
        key = (
            size,
            tuple(snapshot["q"]),
            id(snapshot.get("plan")),
            repr(snapshot.get("boxes", [])),
            repr(snapshot.get("paths", [])),
            repr(snapshot["config"]),
            repr(snapshot.get("discard_points")),
            self.view.yaw,
            self.view.pitch,
            self.view.zoom,
            tuple(self.view.pan),
            self.envelopes.get(),
            self.phase.get(),
        )
        if key != self.last_render_key:
            image = render_scene(
                snapshot, self.view, size, self.envelopes.get(), self.phase.get()
            )
            self.image.image = ImageTk.PhotoImage(image)
            self.image.configure(image=self.image.image)
            self.last_render_key = key
        self.window.after(33, self.refresh)
