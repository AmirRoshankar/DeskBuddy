"""Refresh scene estimates without turning detector revisions into motion faults."""

import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np

from .execution import check_swept_obstacles, require_same_scene, same_geometry


class ScenePathBlocked(RuntimeError):
    pass


class SceneMonitor:
    def __init__(
        self, arm, plan, route, read_scene, notify=lambda _: None, clock=time.monotonic
    ):
        self.arm, self.plan, self.route = arm, plan, route
        self.read_scene, self.notify, self.clock = read_scene, notify, clock
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scene-check"
        )
        self.future = None
        self.pending_boxes = None
        self.pending_times = None
        self.checked_boxes = deepcopy(route["obstacles_m"])
        self.last_poll = -float("inf")
        self.last_good = clock()

    def poll(self, current, phase):
        now = self.clock()
        if now - self.last_poll < 0.05:
            return
        self.last_poll = now
        try:
            live = self.read_scene()
        except (OSError, ValueError):
            if now - self.last_good > 2:
                raise RuntimeError(
                    "Scene monitoring unavailable; position hold retained"
                )
            return  # keep the last scene through a transient snapshot read failure
        require_same_scene({**self.route, "allow_object_collisions": True}, live)
        if (
            "scene_revision" not in (live.get("obstacle_detection") or {})
            or "obstacle_boxes" not in live
        ):
            if now - self.last_good > 2:
                raise RuntimeError("Object map unavailable; position hold retained")
            return
        self.last_good = now
        if self.route.get("allow_object_collisions"):
            return
        actual = deepcopy(live.get("obstacle_boxes", []))
        if not same_geometry(self.checked_boxes, actual):
            self.route["obstacles_m"] = actual
        self.route["scene_revision"] = (live.get("obstacle_detection") or {}).get(
            "scene_revision"
        )
        if self.future is not None and self.future.done():
            # A result for an obsolete geometry version must not stop this route.
            if same_geometry(self.pending_boxes, actual, tolerance=1e-6):
                error = self.future.exception()
                if error is not None:
                    if isinstance(error, ValueError):
                        fraction = getattr(error, "path_fraction", None)
                        collision_time = (
                            float("inf")
                            if fraction is None
                            else np.interp(
                                fraction * (len(self.pending_times) - 1),
                                np.arange(len(self.pending_times)),
                                self.pending_times,
                            )
                        )
                        if collision_time >= phase - 0.05:
                            raise ScenePathBlocked(
                                f"Updated object blocks remaining path: {error}"
                            ) from error
                    else:
                        raise error
                else:
                    self.checked_boxes = deepcopy(actual)
                    self.notify("Scene updated · remaining path clear")
            self.future = None
        if self.future is None and not same_geometry(self.checked_boxes, actual):
            index = int(np.searchsorted(self.plan.time_s, phase, side="right"))
            remaining = np.vstack([current, self.plan.joints_rad[index:]])
            self.pending_boxes = deepcopy(actual)
            self.pending_times = np.r_[phase, self.plan.time_s[index:]]
            self.future = self.executor.submit(
                check_swept_obstacles, self.arm, remaining, deepcopy(actual)
            )

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
