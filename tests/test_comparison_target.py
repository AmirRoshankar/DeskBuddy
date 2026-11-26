from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from deskbuddy.config import DEFAULT
from deskbuddy.demo import T_BASE_CAMERA
from deskbuddy.overlays import project
from deskbuddy.ui.comparison import ComparisonWindow
from deskbuddy.ui.viewport import CameraViewport


def test_letterboxed_click_stays_on_paper_and_height_is_a_separate_projection():
    config = deepcopy(DEFAULT)
    config["T_base_camera"] = T_BASE_CAMERA.tolist()
    config["test_bed"] = dict(
        corners_base_m=[
            [0.20, -0.10, 0],
            [0.41, -0.10, 0],
            [0.41, 0.197, 0],
            [0.20, 0.197, 0],
        ],
        inset_m=0.005,
    )
    intr = dict(fx=600, fy=600, ppx=320, ppy=240)
    view = ComparisonWindow.__new__(ComparisonWindow)
    view.snapshot = (np.zeros((480, 640, 3), np.uint8), None, intr, config, None, [])
    view.viewport = CameraViewport(1000, 700)
    view.message = SimpleNamespace(set=lambda _: None)
    view.hover = SimpleNamespace(value=0.06, set_limits=lambda *args: None)
    view.clear_results = lambda _: None
    x, y = view.viewport.from_camera(385, 195)
    view.click(SimpleNamespace(x=x + 0.01, y=y + 0.01))
    anchor = project(view.surface, config["T_base_camera"], intr)
    assert np.allclose(anchor, [385, 195], atol=1)
    raised = project(
        view.surface + [0, 0, view.hover.value], config["T_base_camera"], intr
    )
    assert np.linalg.norm(np.array(anchor) - raised) > 1
    assert np.allclose(view.goal, view.surface[:2])


def test_height_change_invalidates_old_comparison_results():
    view = ComparisonWindow.__new__(ComparisonWindow)
    view.version = 7
    view.results = [dict(method="imf")]
    view.cells = {}
    messages = []
    view.message = SimpleNamespace(set=messages.append)
    view.change_height(0.08)
    assert view.version == 8 and not view.results
    assert "Height changed" in messages[-1]


def test_comparison_lifts_to_selected_height_and_reports_whole_arm_rejection():
    from deskbuddy.kinematics import Arm
    from deskbuddy.ui.comparison import full_arm_results

    config = deepcopy(DEFAULT)
    arm = Arm()
    q = np.deg2rad(config["preview_start_deg"])
    target = np.array([0.27, 0.02, 0.12])
    path = np.linspace(arm.fk(q)[:2], target[:2], 32).tolist()
    row = dict(
        method="mlp", valid_candidates=1, candidates=1, raw_paths=[path], total_ms=0.0
    )
    results = full_arm_results([deepcopy(row)], arm, q, target, config, [], 1)
    assert results[0]["valid_candidates"] == 1
    assert np.allclose(results[0]["selected_tip_path_m"][-1], target, atol=0.002)
    # Same visually valid XY path, but the lower target fails the physical 3D checks.
    lower = target.copy()
    lower[2] = -0.005
    results = full_arm_results([deepcopy(row)], arm, q, lower, config, [], 1)
    assert results[0]["xy_valid_candidates"] == 1
    assert results[0]["valid_candidates"] == 0
    assert results[0]["full_arm_rejections"][0]
