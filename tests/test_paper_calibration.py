import numpy as np
import pytest

from deskbuddy.config import load_config
from deskbuddy.paper_calibration import apply_paper, fit_paper

cv2 = pytest.importorskip("cv2")


def fixture():
    intr = dict(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    obj = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])
    k = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    pixels = cv2.projectPoints(
        obj, np.array([0.0, 0, 0.2]), np.array([-0.105, -0.1485, 0.6]), k, None
    )[0].reshape(-1, 2)
    transform = np.diag([1.0, -1, -1, 1])
    transform[:3, 3] = [0.25, 0, 0.7]
    return pixels, intr, transform


def test_known_a4_dimensions_and_pose_define_inset_bounds_without_enabling_go():
    pixels, intr, transform = fixture()
    bed = fit_paper(pixels, intr, transform, np.full((480, 640), 0.6))
    p = np.array(bed["corners_base_m"])
    assert np.linalg.norm(p[1] - p[0]) == pytest.approx(0.210)
    assert np.linalg.norm(p[3] - p[0]) == pytest.approx(0.297)
    assert p[:, 2] == pytest.approx(np.full(4, 0.1))
    assert max(bed["reprojection_error_px"]) < 1e-6
    config = load_config()
    updated = apply_paper(config, bed)
    assert updated["test_bed"] and not updated["calibration_verified"]
    assert config["test_bed"] is None
    assert updated["workspace_min_m"][2] > updated["table_z_m"]


def test_crossed_corners_rejected_but_inconsistent_depth_is_a_warning():
    pixels, intr, transform = fixture()
    with pytest.raises(ValueError, match="without crossing"):
        fit_paper(pixels[[0, 2, 1, 3]], intr, transform)
    bed = fit_paper(pixels, intr, transform, np.full((480, 640), 0.8))
    assert bed["validation_warnings"]
    assert bed["depth_check_samples"] >= 12
    # Missing depth is allowed: known dimensions still constrain the pose.
    assert fit_paper(pixels, intr, transform, np.zeros((480, 640)))


def test_flat_paper_with_large_robot_frame_slope_is_accepted_and_used_locally():
    from deskbuddy.planning import hover_bounds
    from deskbuddy.test_bed import table_clearance, table_height_at

    _, intr, transform = fixture()
    obj = np.array([[0.0, 0, 0], [0.210, 0, 0], [0.210, 0.297, 0], [0, 0.297, 0]])
    k = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    pixels = cv2.projectPoints(
        obj, np.array([0.35, 0.1, 0.2]), np.array([-0.105, -0.1485, 0.6]), k, None
    )[0].reshape(-1, 2)
    bed = fit_paper(pixels, intr, transform)
    points = np.array(bed["corners_base_m"])
    assert np.ptp(points[:, 2]) > 0.03
    config = apply_paper(load_config(), bed)
    for point in points:
        assert table_height_at(point[:2], config) == pytest.approx(point[2] + 0.003)
        assert table_clearance(point + [0, 0, 0.025], config) == pytest.approx(0.007)
    center = points.mean(0)
    assert hover_bounds(center, config)[0] == pytest.approx(0.02)


def test_trajectory_clearance_follows_plane_instead_of_highest_corner():
    from deskbuddy.planning import Plan, validate_plan

    config = load_config()
    config.update(workspace_min_m=[0, -0.1, 0], workspace_max_m=[0.3, 0.4, 0.5])
    config["test_bed"] = dict(
        corners_base_m=[
            [0, 0, 0],
            [0.21, 0, 0.105],
            [0.21, 0.297, 0.105],
            [0, 0.297, 0],
        ],
        inset_m=0.02,
    )

    class Arm:
        lower, upper = np.full(5, -1.0), np.full(5, 1.0)

        def fk(self, q):
            x = 0.05 + q[0]
            return np.array([x, 0.1, 0.5 * x + 0.025])

        def frames(self, q):
            return np.tile(self.fk(q), (6, 1)), None

    arm = Arm()
    q = np.zeros((3, 5))
    q[:, 0] = [0, 0.05, 0.1]
    plan = Plan(np.array([0.0, 1.0, 2.0]), q, arm.fk(q[-1]), 0.0)
    assert validate_plan(arm, plan, config) is plan


def test_corner_depth_contamination_does_not_veto_correct_paper():
    pixels, intr, transform = fixture()
    depth = np.full((480, 640), 0.6)
    for u, v in pixels:
        x, y = int(round(u)), int(round(v))
        depth[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5] = 0.9
    bed = fit_paper(pixels, intr, transform, depth)
    assert not bed["validation_warnings"]
    assert max(bed["depth_error_mm"]) < 1e-5


def test_known_horizontal_base_plane_is_enforced_without_depth_veto():
    pixels, intr, transform = fixture()
    bed = fit_paper(
        pixels, intr, transform, np.full((480, 640), 0.8), table_z=0.0, strict=False
    )
    assert np.allclose(np.array(bed["corners_base_m"])[:, 2], 0.0)
    assert bed["validation_warnings"]


def test_level_table_click_uses_same_a4_correspondences_as_second_camera():
    from deskbuddy.test_bed import surface_point

    pixels, intr, transform = fixture()
    bed = fit_paper(pixels, intr, transform, table_z=0.0, strict=False)
    config = load_config()
    config.update(
        test_bed=bed,
        assume_level_table=True,
        table_z_m=0.0,
        T_base_camera=transform.tolist(),
        restrict_to_test_bed=False,
    )
    for pixel, point in zip(pixels, bed["corners_base_m"]):
        assert np.allclose(surface_point(*pixel, intr, config), point, atol=1e-6)
