from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.overlays import grid_segments
from deskbuddy.planning import check_workspace
from deskbuddy.test_bed import contains_xy, corners, surface_point


def config():
    c = deepcopy(DEFAULT)
    c["T_base_camera"] = np.eye(4).tolist()
    c["test_bed"] = {
        "corners_base_m": [
            [0, 0, 0.5],
            [0.21, 0, 0.5],
            [0.21, 0.297, 0.5],
            [0, 0.297, 0.5],
        ],
        "inset_m": 0.02,
    }
    return c


def test_known_rectangle_inset_rejects_edges_and_maps_pixels_to_surface():
    c = config()
    intr = {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240}
    assert np.allclose(corners(c)[0], [0.02, 0.02, 0.5])
    assert contains_xy([0.1, 0.1, 0.7], c)
    assert not contains_xy([0.005, 0.1, 0.7], c)
    assert np.allclose(surface_point(440, 360, intr, c), [0.1, 0.1, 0.5])
    with pytest.raises(ValueError, match="inset"):
        surface_point(320, 240, intr, c)
    assert len(grid_segments(c, intr)) == 10


def test_backend_rejects_outside_paper_even_inside_workspace_box():
    c = config()
    c["workspace_min_m"] = [0, 0, 0.04]
    c["workspace_max_m"] = [0.5, 0.5, 1]
    check_workspace([0.1, 0.1, 0.6], c)
    with pytest.raises(ValueError, match="A4"):
        check_workspace([0.3, 0.1, 0.6], c)
    c["test_bed"]["corners_base_m"][2][0] += 0.02
    with pytest.raises(ValueError, match="rectangular"):
        corners(c)
