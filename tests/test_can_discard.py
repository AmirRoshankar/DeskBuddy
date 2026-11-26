import numpy as np
import pytest

from deskbuddy.can_discard import (
    detect_can,
    inside_zone,
    jaw_geometry,
    opening_for_width,
    plan_can_cycle,
    run_transfer,
    zone_points,
)
from deskbuddy.config import load_config
from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.perception import DetectedBox


def fixture():
    c = load_config()
    c.update(
        workspace_min_m=[0.04, -0.3, 0.02],
        workspace_max_m=[0.5, 0.3, 0.4],
        restrict_to_test_bed=False,
        discard_zone=[
            [0.22, -0.12, 0],
            [0.30, -0.12, 0],
            [0.30, -0.04, 0],
            [0.22, -0.04, 0],
        ],
    )
    return c, dict(center_m=[0.25, 0.04, 0], diameter_m=0.066, height_m=0.122)


def test_zone_is_ordered_and_nominal_can_fits_opening():
    c, _ = fixture()
    zone = zone_points(c["discard_zone"])
    assert inside_zone([0.26, -0.08], zone)
    assert not inside_zone([0.5, -0.08], zone)
    with pytest.raises(ValueError):
        zone_points(zone[[0, 2, 1, 3]])
    arm = Arm()
    percent = opening_for_width(arm, 0.072)
    assert 0 < percent < 100 and jaw_geometry(arm, percent)[1] >= 0.072


def test_whole_can_cycle_is_preflighted_and_drop_tip_inside_zone():
    c, can = fixture()
    arm = Arm()
    q = np.deg2rad(c["preview_start_deg"])
    plan, job = plan_can_cycle(arm, q, c, can)
    req = job["can_request"]
    assert len(req["phases"]) == 4
    assert req["close_percent"] < req["open_percent"]
    assert inside_zone(plan.target_m, c["discard_zone"])
    assert np.min([arm.fk(p)[2] for p in plan.joints_rad]) >= c["workspace_min_m"][2]
    assert c["can_request"] is None


def test_can_transfer_orders_close_lift_drop_and_stop_never_releases():
    calls = []
    run_transfer(
        lambda i: calls.append(("move", i)),
        lambda p: calls.append(("grip", p)),
        lambda: False,
        55,
        49,
    )
    assert calls == [
        ("grip", 55),
        ("move", 0),
        ("move", 1),
        ("grip", 49),
        ("move", 2),
        ("move", 3),
        ("grip", 55),
    ]
    calls = []
    with pytest.raises(RuntimeError):
        run_transfer(
            lambda i: calls.append(("move", i)),
            lambda p: calls.append(("grip", p)),
            lambda: len(calls) >= 5,
            55,
            49,
        )
    assert calls[-1] == ("move", 2) and calls.count(("grip", 55)) == 1


def test_flat_paper_is_not_a_can_target():
    c, _ = fixture()
    paper = DetectedBox(
        np.array([0.2, 0, 0]),
        np.array([0.4, 0.2, 0.002]),
        1000,
        0.002,
        (0, 0, 100, 100),
    )
    with pytest.raises(ValueError, match="No upright can"):
        detect_can([paper], c)


def test_worker_executes_preflighted_phases_with_injected_visual_pickup_evidence(
    monkeypatch,
):
    from types import SimpleNamespace

    from deskbuddy.can_discard import execute_can
    from deskbuddy.kinematics import JOINTS

    c, can = fixture()
    arm = Arm()
    q = np.deg2rad(c["preview_start_deg"])
    selected, cfg = plan_can_cycle(arm, q, c, can)
    actual = [q.copy()]
    calls = []

    class Verifier:
        def __init__(self, *args):
            pass

        def check(self, *args):
            return True

    monkeypatch.setattr("deskbuddy.can_pickup.PickupVerifier", Verifier)

    def reach(bus, target, config, stopped, publish, **kwargs):
        plan = kwargs["selected_plan"]
        actual[0] = plan.joints_rad[-1].copy()
        calls.append("move")
        return plan, dict(completed=True)

    monkeypatch.setattr("deskbuddy.execute_reach.execute_target", reach)
    monkeypatch.setattr(
        "deskbuddy.gripper_control.ramp_gripper",
        lambda bus, percent, stopped: calls.append(percent),
    )
    bus = SimpleNamespace(
        sync_read=lambda *args, **kwargs: dict(zip(JOINTS, np.rad2deg(actual[0])))
    )
    _, result = execute_can(bus, selected, cfg, lambda: False, lambda report: None)
    assert calls.count("move") == 4
    assert result["can_discard_completed"] and result["pickup_verified"]
    assert not result["drop_verified"]
    assert calls[-1] == cfg["can_request"]["open_percent"]


def test_known_radius_fit_recovers_center_from_depth_instead_of_biased_box_center():
    c, _ = fixture()
    c["T_base_camera"] = np.diag([1.0, -1.0, -1.0, 1.0]).tolist()
    c["T_base_camera"][0][3] = 0.25
    c["T_base_camera"][1][3] = 0.04
    c["T_base_camera"][2][3] = 0.6
    intr = dict(fx=500.0, fy=500.0, ppx=100.0, ppy=100.0)
    depth = np.zeros((200, 200))
    for height in np.linspace(0.03, 0.099, 30):
        angle = np.linspace(0, 2 * np.pi, 600)
        z = 0.6 - height
        u = np.rint(0.033 * np.cos(angle) / z * 500 + 100).astype(int)
        v = np.rint(-0.033 * np.sin(angle) / z * 500 + 100).astype(int)
        for x, y in zip(u, v):
            if depth[y, x] == 0 or z < depth[y, x]:
                depth[y, x] = z
    box = DetectedBox(
        np.array([0.23, 0.015, 0]),
        np.array([0.29, 0.08, 0.122]),
        1000,
        0.122,
        (60, 60, 140, 140),
    )
    can = detect_can([box], c, depth, intr)
    assert "cylinder fit" in can["source"]
    assert np.linalg.norm(np.array(can["center_m"][:2]) - [0.25, 0.04]) < 0.002


def test_rim_training_targets_are_on_lip_with_radial_closing_axis():
    import torch

    from deskbuddy.learned.grasp_lab import can_rim_data, geometric_metrics

    context, target, groups = can_rim_data(25)
    assert len(target) == 100 and len(set(groups)) == 25
    valid = geometric_metrics(
        torch.tensor(target), torch.tensor(context), True, profile="can_rim"
    )
    assert valid.all()
    np.testing.assert_allclose(
        target[:, 2] * 0.05, context[:, 2] * 0.05 - 0.002, atol=1e-7
    )
    assert np.all((target[:, 7] * 0.05 >= 0.007) & (target[:, 7] * 0.05 <= 0.009))


def test_rim_model_samples_valid_lip_proposals_without_body_fallback():
    from pathlib import Path

    from deskbuddy.learned.grasp_proposals import propose

    checkpoint = "runs/flowreach/can-rim-grasps-v1/imf.pt"
    if not Path(checkpoint).exists():
        pytest.skip("Offline rim experiment checkpoint is not distributed")
    rows = propose([0.066, 0.066, 0.12222], candidates=32, checkpoint=checkpoint)
    valid = [row for row in rows if row["geometric_valid"]]
    assert len(valid) >= 20
    assert all(row["grasp_mode"] == "rim" for row in rows)
    assert all(
        abs(np.linalg.norm(row["offset_m"][:2]) - 0.033) < 0.002 for row in valid
    )
    assert all(abs(row["offset_m"][2] - 0.12022) < 0.002 for row in valid)


def test_can_height_uses_physical_plane_not_paper_clearance_margin():
    c, _ = fixture()
    c["test_bed"] = dict(
        corners_base_m=[
            [0.1, -0.1, 0],
            [0.31, -0.1, 0],
            [0.31, 0.197, 0],
            [0.1, 0.197, 0],
        ],
        clearance_margin_m=0.003,
    )
    box = DetectedBox(
        np.array([0.2, 0, 0]),
        np.array([0.266, 0.066, 0.12222]),
        500,
        0.12222,
        (0, 0, 20, 30),
    )
    can = detect_can([box], c)
    assert can["center_m"][2] == pytest.approx(0)
    assert can["height_m"] == pytest.approx(0.12222)
