import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deskbuddy.geometry import (
    deproject,
    fit_rigid,
    median_depth,
    transform_point,
    validate_transform,
)


def test_deprojection_off_center_and_units():
    assert np.allclose(
        deproject(420, 190, 0.6, (600, 600, 320, 240)), [0.1, -0.05, 0.6]
    )
    with pytest.raises(ValueError):
        deproject(1, 2, float("nan"), (600, 600, 320, 240))


def test_depth_zeros_outliers_edges_and_missing():
    depth = np.full((10, 10), 0.65)
    depth[4, 4] = 0
    depth[4, 5] = 1.9
    assert median_depth(depth, 5, 5) == pytest.approx(0.65)
    with pytest.raises(ValueError):
        median_depth(np.zeros((10, 10)), 5, 5)
    with pytest.raises(ValueError):
        median_depth(depth, -1, 5)
    depth[:, 5:] = 0.9
    with pytest.raises(ValueError, match="discontinuity"):
        median_depth(depth, 5, 5)


def test_rigid_fit_recovers_direction_with_planar_points():
    camera = np.array([[0, 0, 0.65], [0.1, 0, 0.65], [0, 0.2, 0.65], [0.1, 0.2, 0.65]])
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.8]).as_matrix()
    t[:3, 3] = [0.15, 0.03, 0.4]
    base = camera @ t[:3, :3].T + t[:3, 3]
    fitted, errors = fit_rigid(camera, base)
    assert np.allclose(fitted, t, atol=1e-10)
    assert errors.max() < 1e-10
    assert np.allclose(transform_point(fitted, camera[1]), base[1])


def test_collinear_and_reflected_transform_rejected():
    with pytest.raises(ValueError, match="collinear"):
        fit_rigid(np.arange(12).reshape(4, 3), np.arange(12).reshape(4, 3))
    t = np.eye(4)
    t[0, 0] = -1
    with pytest.raises(ValueError):
        validate_transform(t)
