import numpy as np
from PIL import Image
from test_execute_reach import setup

from deskbuddy.kinematics import Arm
from deskbuddy.overlays import project
from deskbuddy.trajectory.path_overlays import COLORS, OverlayPlanner, draw_paths


def test_all_four_methods_have_distinct_paths_and_colors_without_execution():
    config, target = setup()
    arm = Arm()
    start = np.deg2rad(config["preview_start_deg"])

    class Suite:
        def compare(self, scene, a, b, *args, **kwargs):
            results = []
            for method, bend in [("mlp", 0.004), ("imf", -0.004)]:
                path = np.linspace(a, b, 32)
                path[:, 1] += np.sin(np.linspace(0, np.pi, 32)) * bend
                results.append(dict(method=method, raw_paths=[path.tolist()]))
            return results

    planner = OverlayPlanner()
    planner.suite = Suite()
    entries = planner.compare(arm, start, target, config, [], 1, 1)
    assert {e["method"] for e in entries} == set(COLORS)
    assert len({e["color"] for e in entries}) == 4
    assert all(e["points_m"] is not None for e in entries)
    image = Image.new("RGB", (640, 480), "#171719")
    from deskbuddy.demo import T_BASE_CAMERA

    intr = dict(fx=600, fy=600, ppx=320, ppy=240)
    draw_paths(image, entries, lambda p: project(p, T_BASE_CAMERA, intr), "rrt")
    assert len(image.getcolors(maxcolors=100000)) >= 3
