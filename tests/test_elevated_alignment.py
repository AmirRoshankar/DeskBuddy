import numpy as np
import pytest

from deskbuddy.elevated_alignment import check_elevated_tip


def setup():
    intrinsics = dict(fx=100, fy=100, ppx=50, ppy=50)
    result = dict(T_base_camera=np.eye(4).tolist(), intrinsics=intrinsics)
    return result, intrinsics, dict(T_base_camera=np.eye(4).tolist())


def test_missing_depth_identifies_pair_not_more_points():
    r, i, c = setup()
    with pytest.raises(ValueError, match="P1: no saved depth"):
        check_elevated_tip(
            r, [dict(primary_pixel=[50, 50], secondary_pixel=[50, 50])], i, c
        )


def test_bad_old_pair_does_not_block_three_valid_pairs():
    r, i, c = setup()
    points = [dict(primary_pixel=[50, 50], secondary_pixel=[50, 50])]
    points += [
        dict(
            primary_pixel=[u, 50],
            secondary_pixel=[u, 50],
            depth_patch=np.full((5, 5), 0.5).tolist(),
        )
        for u in (30, 50, 70)
    ]
    result = check_elevated_tip(r, points, i, c)
    assert result["passed"]
    assert result["point_ids"] == [2, 3, 4]
    assert "4 pairs captured; 3/3 usable" in result["status"]
    assert "P1: no saved depth" in result["status"]


def test_mismatch_reports_worst_pair_and_pixels():
    r, i, c = setup()
    points = [
        dict(
            primary_pixel=[u, 50],
            secondary_pixel=[u + 20, 50],
            depth_patch=np.full((5, 5), 0.5).tolist(),
        )
        for u in (30, 50, 70)
    ]
    result = check_elevated_tip(r, points, i, c)
    assert not result["passed"]
    assert "20.0 px (max 8)" in result["status"]
