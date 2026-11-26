from types import SimpleNamespace

import numpy as np
import pytest

from deskbuddy.config import load_config
from deskbuddy.kinematics import Arm
from deskbuddy.ui.scene_inspector import (
    OrbitView,
    SceneNavigation,
    pick_on_plane,
    scene_projection,
)


def test_3d_click_inverse_uses_same_projection_with_pan_zoom_and_rotation():
    snapshot = dict(arm=Arm(), q=np.deg2rad([0, -15, 30, 15, 0]), config=load_config())
    view = OrbitView()
    view.rotate(80, -20)
    view.pan += [20, -30]
    view.scroll(240)
    size = (800, 600)
    center, scale = scene_projection(snapshot, size)
    point = np.array([0.25, 0.04, 0.08])
    pixel = view.project([point], center, scale, size)[0]
    assert np.allclose(pick_on_plane(snapshot, view, size, pixel, 0.08), point)


def test_navigation_requires_ctrl_and_plain_click_picks_target():
    events = []
    label = SimpleNamespace(bind=lambda *args: None)
    nav = SceneNavigation(
        label,
        on_pick=lambda e: events.append("pick"),
        on_hover=lambda e: events.append("hover"),
    )
    event = SimpleNamespace(x=10, y=10, state=0, delta=120)
    nav.begin(event, 1)
    nav.release(event)
    nav.wheel(event)
    assert events == ["pick", "hover"]
    event.state = 4
    nav.begin(event, 1)
    before = nav.view.yaw
    nav.move(SimpleNamespace(x=30, y=20, state=4), 1)
    assert nav.view.yaw != before
    nav.release(event)
    assert events == ["pick", "hover"]


def test_motion_events_are_coalesced_and_latest_pan_applied_without_backlog():
    callbacks = []
    draws = []
    label = SimpleNamespace(
        bind=lambda *a: None,
        after=lambda ms, fn: callbacks.append(fn) or "pending",
        winfo_exists=lambda: True,
    )
    nav = SceneNavigation(label, changed=lambda: draws.append(nav.view.pan.copy()))
    nav.begin(SimpleNamespace(x=0, y=0, state=4), 3)
    for x in range(1, 101):
        nav.move(SimpleNamespace(x=x, y=20, state=4), 3)
    assert len(callbacks) == 1 and not draws
    callbacks.pop()()
    assert np.array_equal(draws[0], [100, 20])


def test_navigation_keeps_stable_frame_and_preserves_screen_directions():
    snapshot = dict(arm=Arm(), q=np.deg2rad([0, -15, 30, 15, 0]), config=load_config())
    view = OrbitView()
    size = (800, 600)
    center, scale = scene_projection(snapshot, size, view)
    point = np.array([0.2, 0.03, 0.1])
    before = view.project([point], center, scale, size)[0]
    view.pan += [40, 25]
    assert np.allclose(view.project([point], center, scale, size)[0] - before, [40, 25])
    snapshot["q"] = np.zeros(5)
    next_center, next_scale = scene_projection(snapshot, size, view)
    assert np.array_equal(next_center, center) and next_scale == scale
    assert (
        view.project([point + [0, 0, 0.05]], center, scale, size)[0, 1]
        < view.project([point], center, scale, size)[0, 1]
    )
    view.rotate(0, -10000)
    assert view.pitch >= 0.1
    view.rotate(0, 10000)
    assert view.pitch < np.pi / 2


def test_panning_does_not_repeat_fk_route_or_self_collision_work(monkeypatch):
    from deskbuddy import self_collision
    from deskbuddy.planning import Plan
    from deskbuddy.ui.scene_inspector import render_scene

    arm = Arm()
    q = np.deg2rad([0, -15, 30, 15, 0])
    config = load_config()
    plan = Plan(
        np.linspace(0, 1, 20), np.linspace(q, q + 0.01, 20), arm.fk(q + 0.01), 0.0
    )
    snapshot = dict(arm=arm, q=q, config=config, plan=plan)
    hits = []
    original = self_collision.collisions
    monkeypatch.setattr(
        self_collision, "collisions", lambda *args: hits.append(1) or original(*args)
    )
    view = OrbitView()
    render_scene(snapshot, view, (640, 480), False)
    route = view.geometry.route
    for _ in range(5):
        view.pan += [2, 3]
        render_scene(snapshot, view, (640, 480), False)
    assert len(hits) == 1 and view.geometry.route is route
    snapshot["q"] = q + 0.002
    render_scene(snapshot, view, (640, 480), False)
    assert len(hits) == 2


def test_click_inside_projected_box_does_not_extrapolate_outside_at_hover_height():
    from deskbuddy.ui.scene_inspector import pick_in_workspace

    snapshot = dict(arm=Arm(), q=np.zeros(5), config=load_config())
    snapshot["config"].update(
        workspace_min_m=[-0.2376, -0.2359, 0.0124], workspace_max_m=[0.402, 0.340, 0.3]
    )
    view = OrbitView()
    size = (800, 600)
    center, scale = scene_projection(snapshot, size, view)
    visible_point = np.array([0.1, -0.2359, 0.05])
    pixel = view.project([visible_point], center, scale, size)[0]
    old = pick_on_plane(snapshot, view, size, pixel, 0.1315)
    assert old[1] < snapshot["config"]["workspace_min_m"][1]
    selected = pick_in_workspace(snapshot, view, size, pixel, 0.1315)
    assert np.all(selected >= snapshot["config"]["workspace_min_m"])
    assert np.all(selected <= snapshot["config"]["workspace_max_m"])
    assert np.allclose(view.project([selected], center, scale, size)[0], pixel)
    with pytest.raises(ValueError):
        pick_in_workspace(snapshot, view, size, [-10000, -10000], 0.1315)
