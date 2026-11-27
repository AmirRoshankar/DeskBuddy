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


def test_rim_proposals_are_moved_from_body_radius_onto_the_lid_seam():
    from deskbuddy.can_discard import LID_DIAMETER_M, to_lid_radius

    can = dict(diameter_m=0.066, height_m=0.12222)
    angle = 0.7
    offset, shift = to_lid_radius(
        [0.033 * np.cos(angle), 0.033 * np.sin(angle), 0.1202], can
    )
    assert shift == pytest.approx((0.066 - LID_DIAMETER_M) / 2)
    assert np.linalg.norm(offset[:2]) == pytest.approx(LID_DIAMETER_M / 2)
    assert np.arctan2(offset[1], offset[0]) == pytest.approx(angle)
    assert offset[2] == pytest.approx(0.1202)


def test_hole_proposal_puts_one_tip_in_the_hole_and_one_outside_the_wall():
    from deskbuddy.can_discard import (
        HOLE_EXTRA_OPENING_M,
        HOLE_OUTSIDE_CLEARANCE_M,
        hole_proposal,
    )

    can = dict(center_m=[0.28, -0.03, 0.0], diameter_m=0.066, height_m=0.12222)
    opening = dict(
        lid_center_m=[0.28, -0.03, 0.12222],
        center_m=[0.28 + 0.017, -0.03, 0.11622],
        confidence=0.6,
    )
    proposal = hole_proposal(can, opening)
    middle = np.array(can["center_m"][:2]) + proposal["offset_m"][:2]
    half = (proposal["opening_m"] - HOLE_EXTRA_OPENING_M) / 2
    w, x, y, z = proposal["quaternion_wxyz"]
    closing = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    tips = [middle - closing * half, middle + closing * half]
    radii = sorted(np.linalg.norm(t - [0.28, -0.03]) for t in tips)
    assert radii[0] == pytest.approx(0.017, abs=1e-6)
    assert radii[1] == pytest.approx(0.033 + HOLE_OUTSIDE_CLEARANCE_M, abs=1e-6)
    assert proposal["offset_m"][2] < can["height_m"]


def test_lid_vote_prefers_the_hypothesis_whose_ellipse_matches_image_edges():
    import cv2

    from deskbuddy.can_pickup import lid_vote

    angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ring = np.c_[200 + 40 * np.cos(angle), 150 + 15 * np.sin(angle)]
    lifted = ring + [0, -25]

    def image(points):
        out = np.zeros((300, 400, 3), np.uint8)
        cv2.polylines(out, [np.int32(points)], True, (220, 220, 220), 2)
        return out

    assert lid_vote(image(lifted), ring, lifted) is True
    assert lid_vote(image(ring), ring, lifted) is False
    assert lid_vote(np.zeros((300, 400, 3), np.uint8), ring, lifted) is None


def test_hole_grasp_checks_with_cameras_and_backs_out_to_readjust_a_miss(
    monkeypatch,
):
    from types import SimpleNamespace

    from deskbuddy.can_discard import (
        center_tip_in_hole,
        execute_can,
        fingertips,
        hole_proposal,
        unpack,
    )
    from deskbuddy.kinematics import JOINTS

    c, can = fixture()
    arm = Arm()
    q = np.deg2rad(c["preview_start_deg"])
    center = np.asarray(can["center_m"])
    opening = dict(
        lid_center_m=[*center[:2], can["height_m"]],
        center_m=[center[0] + 0.017, center[1], can["height_m"] - 0.006],
        confidence=0.7,
    )
    proposal = center_tip_in_hole(arm, q, can, hole_proposal(can, opening))
    selected, cfg = plan_can_cycle(arm, q, c, can, proposal=proposal)
    cfg["can_request"]["grasp_mode"] = "hole"
    down = unpack(cfg["can_request"]["phases"][1]).joints_rad
    tip = fingertips(arm, down[-1], cfg["can_request"]["open_percent"])
    assert min(np.linalg.norm(t[:2] - opening["center_m"][:2]) for t in tip) < 5e-4

    actual = [q.copy()]
    methods = []

    class Verifier:
        def __init__(self, *args):
            pass

        def check(self, *args):
            return True

    monkeypatch.setattr("deskbuddy.can_pickup.PickupVerifier", Verifier)
    monkeypatch.setattr(
        "deskbuddy.can_pickup.read_images",
        lambda read_live, newer=0: (1.0, [np.zeros((4, 4, 3))], dict(intrinsics={})),
    )
    misses = iter(
        [
            dict(applied=False, reason="test"),
            dict(applied=False, relative_error_m=[0.005, 0.0]),
            dict(applied=False, relative_error_m=[0.0, 0.0]),
            dict(applied=False, relative_error_m=[0.0, 0.0]),
        ]
    )

    def measure(images, intrinsics, config, arm_, pose, percent, can_, debug):
        report = dict(next(misses))
        if "relative_error_m" in report:
            # Report the error relative to wherever the plan now aims.
            fixed, moving = fingertips(arm_, pose, percent)
            tip_ = moving if proposal.get("inside_tip") == "moving" else fixed
            # The hole moves with the (re-aimed) can the plan now targets.
            aim = np.asarray(can_["center_m"][:2]) + hole_offset
            report["relative_error_m"] = (
                np.asarray(report["relative_error_m"]) + aim - tip_[:2]
            ).tolist()
        return report

    hole_offset = np.asarray(opening["center_m"][:2]) - center[:2]
    monkeypatch.setattr("deskbuddy.can_servo.measure", measure)
    monkeypatch.setattr("deskbuddy.can_servo.save_log", lambda *a, **k: "log")

    def reach(bus, target, config, stopped, publish, **kwargs):
        plan = kwargs["selected_plan"]
        actual[0] = plan.joints_rad[-1].copy()
        methods.append(plan.planner_info[0].get("trajectory_method"))
        return plan, dict(completed=True)

    monkeypatch.setattr("deskbuddy.execute_reach.execute_target", reach)
    monkeypatch.setattr(
        "deskbuddy.gripper_control.ramp_gripper", lambda bus, percent, stopped: None
    )
    bus = SimpleNamespace(
        sync_read=lambda *args, **kwargs: dict(zip(JOINTS, np.rad2deg(actual[0])))
    )
    _, result = execute_can(
        bus, selected, cfg, lambda: False, lambda report: None, read_live=lambda: {}
    )
    assert result["can_discard_completed"]
    assert "hole retract" in methods
    assert methods.count("hole descent") >= 3


def test_miss_is_confirmed_by_detection_gap_before_retrying(monkeypatch):
    import time as clock
    from types import SimpleNamespace

    from deskbuddy.can_discard import execute_can
    from deskbuddy.kinematics import JOINTS

    c, can = fixture()
    arm = Arm()
    q = np.deg2rad(c["preview_start_deg"])
    selected, cfg = plan_can_cycle(arm, q, c, can)
    actual = [q.copy()]
    outcomes = iter([None, True])

    class Verifier:
        def __init__(self, *args):
            pass

        def check(self, *args):
            return next(outcomes)

    monkeypatch.setattr("deskbuddy.can_pickup.PickupVerifier", Verifier)
    monkeypatch.setattr(
        "deskbuddy.can_pickup.read_images", lambda read_live, newer=0: None
    )
    center = np.asarray(can["center_m"])

    def read_live():
        # The can is still detected where it was, on the table.
        return dict(
            obstacle_detection=dict(
                detection_unix_s=clock.time() + 1,
                raw_obstacle_boxes=[
                    dict(
                        low_m=[*(center[:2] - 0.033), 0.0],
                        high_m=[*(center[:2] + 0.033), 0.122],
                    )
                ],
            )
        )

    def reach(bus, target, config, stopped, publish, **kwargs):
        plan = kwargs["selected_plan"]
        actual[0] = plan.joints_rad[-1].copy()
        return plan, dict(completed=True)

    monkeypatch.setattr("deskbuddy.execute_reach.execute_target", reach)
    monkeypatch.setattr(
        "deskbuddy.gripper_control.ramp_gripper", lambda bus, percent, stopped: None
    )
    reports = []
    bus = SimpleNamespace(
        sync_read=lambda *args, **kwargs: dict(zip(JOINTS, np.rad2deg(actual[0])))
    )
    _, result = execute_can(
        bus, selected, cfg, lambda: False, reports.append, read_live=read_live
    )
    assert result["can_discard_completed"] and result["can_attempts"] == 2
    detection = next(r["pickup_detection"] for r in reports if "pickup_detection" in r)
    assert all(detection["can_on_table"]) and detection["jaw_lift_mm"] >= 20
