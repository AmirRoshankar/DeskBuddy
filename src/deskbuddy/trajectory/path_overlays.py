"""Four simultaneous, explicitly validated path previews; never executes hardware."""

import copy

import numpy as np
from PIL import ImageDraw

from ..planning import Plan, interpolate_joints, validate_plan
from .execution import check_swept_obstacles, lift_route
from .joint_planner import check_self_collision, plan_joint_route

COLORS = dict(direct="#D4D4D8", mlp="#F4C66A", imf="#C19AFF", rrt="#63D7C4")


class OverlayPlanner:
    def __init__(self):
        self.suite = None

    def compare(
        self, arm, start, target, config, boxes, candidates, revision, on_update=None
    ):
        entries = []

        def changed():
            if on_update is not None:
                on_update(copy.deepcopy(entries))

        geometry = [b.as_dict() if hasattr(b, "as_dict") else b for b in boxes]

        def entry(method, plan=None, points=None, error=None, checked=False):
            valid = error is None
            if plan is not None:
                indices = np.unique(
                    np.linspace(
                        0,
                        len(plan.joints_rad) - 1,
                        min(100, len(plan.joints_rad)),
                        dtype=int,
                    )
                )
                points = np.array([arm.fk(plan.joints_rad[i]) for i in indices])
                try:
                    if not checked:
                        validate_plan(
                            arm, plan, {**config, "allow_object_collisions": False}
                        )
                        check_swept_obstacles(arm, plan.joints_rad, geometry)
                        for pose in plan.joints_rad:
                            check_self_collision(arm, pose)
                except ValueError as exc:
                    valid, error = False, str(exc)
            return dict(
                method=method,
                color=COLORS[method],
                valid=valid,
                points_m=None if points is None else np.asarray(points).tolist(),
                reason=error,
            )

        try:
            goal = arm.ik(
                target, start, config["ik_tolerance_m"], config.get("locked_joints", [])
            )
            times, q = interpolate_joints(start, goal, config)
            entries.append(entry("direct", Plan(times, q, target, 0.0)))
        except ValueError as exc:
            entries.append(entry("direct", error=str(exc)))
        changed()
        try:
            from .live_preview import PlannerSuite, preview_scene

            if self.suite is None:
                self.suite = PlannerSuite()
            results = self.suite.compare(
                preview_scene(config, boxes),
                arm.fk(start)[:2],
                target[:2],
                candidates,
                allow_collision=True,
                methods=("mlp", "imf"),
            )
            for method in ("mlp", "imf"):
                row = next(r for r in results if r["method"] == method)
                trials = []
                for path in row["raw_paths"]:
                    path = np.array(path)
                    try:
                        plan = lift_route(
                            arm,
                            start,
                            target,
                            path,
                            {**config, "allow_object_collisions": False},
                            method=method,
                            candidates=candidates,
                            boxes=boxes,
                            scene_revision=revision,
                        )
                        trials.append(entry(method, plan, checked=True))
                    except ValueError as exc:
                        progress = np.linspace(0, 1, len(path))
                        points = np.c_[
                            path,
                            arm.fk(start)[2] * (1 - progress) + target[2] * progress,
                        ]
                        trials.append(
                            entry(
                                method,
                                points=points,
                                error="Unlifted XY proposal: " + str(exc),
                            )
                        )
                entries.append(
                    next((e for e in trials if e["valid"]), trials[0])
                    if trials
                    else entry(method, error="No proposal")
                )
                changed()
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            for method in ("mlp", "imf"):
                if not any(e["method"] == method for e in entries):
                    entries.append(entry(method, error=str(exc)))
        try:
            plan = plan_joint_route(
                arm, start, target, copy.deepcopy(config), boxes, candidates, revision
            )
            entries.append(
                entry("rrt", plan, checked=not config.get("allow_object_collisions"))
            )
        except ValueError as exc:
            entries.append(entry("rrt", error=str(exc)))
        changed()
        return entries


def draw_paths(image, entries, projection, selected=None):
    """Colored geometry only; the dashboard sidebar owns the legend."""
    from ..motion_overlay import dashed

    draw = ImageDraw.Draw(image)
    for entry in entries:
        if entry["points_m"] is None:
            continue
        color = entry["color"]
        points = [projection(p) for p in entry["points_m"]]
        for a, b in zip(points[:-1], points[1:]):
            if a is None or b is None:
                continue
            if entry["valid"] and not entry.get("stale"):
                if selected == entry["method"]:
                    draw.line((a, b), fill="#171719", width=6)
                draw.line(
                    (a, b), fill=color, width=3 if selected == entry["method"] else 2
                )
            else:
                dashed(draw, a, b, color, width=2, dash=4)
