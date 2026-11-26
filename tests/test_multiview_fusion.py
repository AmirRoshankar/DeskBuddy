import numpy as np

from deskbuddy.trajectory.multiview_evidence import (
    filter_secondary,
    fuse_fragments,
    project_cloud,
)
from deskbuddy.trajectory.perception import DetectedBox


def fixture():
    t = np.eye(4)
    t[2, 3] = -1
    corners = [
        [-0.105, -0.1485, 0],
        [0.105, -0.1485, 0],
        [0.105, 0.1485, 0],
        [-0.105, 0.1485, 0],
    ]
    calibration = dict(
        T_base_camera=t.tolist(),
        intrinsics=dict(fx=1000, fy=1000, ppx=200, ppy=200),
        image_size=[400, 400],
        secondary_pixels=[[95, 51.5], [305, 51.5], [305, 348.5], [95, 348.5]],
        corners_base_m=corners,
        H_secondary_paper=[[1000, 0, 95], [0, 1000, 51.5], [0, 0, 1]],
    )
    config = dict(
        test_bed=dict(corners_base_m=corners, inset_m=0),
        workspace_min_m=[-0.105, -0.1485, 0],
        workspace_max_m=[0.105, 0.1485, 0.2],
    )
    x, y = np.meshgrid(np.linspace(-0.025, 0.025, 20), np.linspace(-0.015, 0.015, 15))
    cloud = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.04)]
    return config, calibration, cloud


def test_webcam_paper_rejected_and_object_clipped_to_raised_support():
    config, calibration, cloud = fixture()
    paper = np.zeros((400, 400), bool)
    paper[52:349, 95:306] = True
    obj = np.zeros_like(paper)
    obj[120:260, 130:270] = True
    instances = [
        dict(mask=paper, roi=[95, 52, 306, 349]),
        dict(mask=obj, roi=[130, 120, 270, 260]),
    ]
    accepted, projected = filter_secondary(instances, calibration, config, cloud)
    assert len(accepted) == 1
    assert accepted[0]["mask"].sum() < obj.sum() / 2
    assert projected is not None


def test_two_partial_boxes_become_one_measured_two_view_object():
    config, calibration, cloud = fixture()
    mask = np.zeros((400, 400), bool)
    mask[175:225, 165:235] = True
    instances = [dict(mask=mask, roi=[165, 175, 235, 225])]
    boxes = [
        DetectedBox(
            np.array([-0.025, -0.015, 0]),
            np.array([-0.002, 0.015, 0.04]),
            100,
            0.04,
            (1, 1, 20, 20),
        ),
        DetectedBox(
            np.array([0.002, -0.015, 0]),
            np.array([0.025, 0.015, 0.04]),
            100,
            0.04,
            (21, 1, 40, 20),
        ),
    ]
    uv, valid = project_cloud(cloud, calibration)
    result = fuse_fragments(boxes, instances, calibration, cloud, uv, (uv, valid))
    assert len(result) == 1
    assert result[0].low[0] <= -0.025 and result[0].high[0] >= 0.025
    assert "Two-view" in result[0].source
    assert not fuse_fragments(boxes, [], calibration, cloud, uv, (uv, valid))
