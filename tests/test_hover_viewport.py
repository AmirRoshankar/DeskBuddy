from copy import deepcopy

import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.planning import hover_bounds
from deskbuddy.ui.hover import wheel_height
from deskbuddy.ui.viewport import CameraViewport


def test_smooth_fractional_wheel_input_and_range_clamps():
    assert wheel_height(0.12, 120) == pytest.approx(0.122)
    assert wheel_height(0.12, 15) == pytest.approx(0.12025)
    assert wheel_height(0.12, -120) == pytest.approx(0.118)
    assert wheel_height(0.199, 120) == 0.20
    assert wheel_height(0.021, -120) == 0.02


@pytest.mark.parametrize("size", [(600, 400), (900, 300), (350, 600)])
def test_resized_camera_clicks_keep_the_same_source_pixel(size):
    view = CameraViewport(*size)
    for u, v in [(0, 0), (320, 240), (639, 479)]:
        xy = view.from_camera(u, v)
        point = view.to_camera(*xy)
        assert abs(point[0] - u) <= 1 and abs(point[1] - v) <= 1
    w, h, x, y = view.geometry
    if y > 0:
        assert view.to_camera(10, 0) is None
    if x > 0:
        assert view.to_camera(0, 10) is None


def test_low_hover_respects_measured_table_clearance():
    config = deepcopy(DEFAULT)
    config["workspace_min_m"][2] = 0.019
    config["table_z_m"] = 0.004
    minimum, maximum = hover_bounds([0.25, 0, -0.01], config)
    assert minimum == pytest.approx(0.031)
    assert maximum == 0.20


def test_hover_stays_inside_same_workspace_ceiling_as_selected_targets():
    from deskbuddy.planning import check_workspace

    config = deepcopy(DEFAULT)
    config["workspace_max_m"][2] = 0.15
    surface = [0.25, 0, 0.05]
    minimum, maximum = hover_bounds(surface, config)
    assert maximum == pytest.approx(0.10)
    for height in (minimum, (minimum + maximum) / 2, maximum):
        check_workspace([surface[0], surface[1], surface[2] + height], config)
    with pytest.raises(ValueError, match="outside workspace"):
        hover_bounds([0.8, 0, 0.05], config)
    with pytest.raises(ValueError, match="hover range"):
        hover_bounds([0.25, 0, 0.16], config)


def test_hover_clamps_before_target_marker_leaves_camera_image():
    import numpy as np

    from deskbuddy.overlays import project, visible_hover_bounds

    transform = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    intrinsics = dict(fx=600, fy=600, ppx=320, ppy=240)
    low, high = visible_hover_bounds(
        [0.5, 0, 0], (0.02, 0.20), transform, intrinsics, (640, 480)
    )
    assert low == 0.02
    assert high == pytest.approx((479 - 16 - 240) * 0.5 / 600)
    assert project([0.5, 0, high], transform, intrinsics)[1] <= 463.000001
    with pytest.raises(ValueError, match="camera view"):
        visible_hover_bounds(
            [0.5, 1, 0], (0.02, 0.20), transform, intrinsics, (640, 480)
        )
