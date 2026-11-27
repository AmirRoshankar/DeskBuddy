import numpy as np

from deskbuddy.local_visual_servo import refine


def test_adapts_wrong_model_and_reduces_image_error_with_xy_steps():
    actual = np.array([[1100.0, 200.0], [-100.0, 900.0]])
    model = np.array([[800.0, 0.0], [0.0, 700.0]])
    position = np.array([0.004, -0.003])
    steps = []

    def observe():
        return dict(error_px=actual @ position, jacobian_px_m=model)

    def move(delta):
        steps.append(delta.copy())
        position[:] += 0.85 * delta
        return 0.85 * delta

    result = refine(observe, move, lambda: False)
    assert result["converged"]
    assert np.linalg.norm(actual @ position) <= 2
    assert all(
        delta.shape == (2,) and np.linalg.norm(delta) <= 0.00200001 for delta in steps
    )
    assert result["travelled_m"] <= 0.01000001
    assert not result["depth_used"]


def test_missing_webcam_measurement_never_guesses_motion():
    steps = []
    report = refine(lambda: None, lambda delta: steps.append(delta), lambda: False)
    assert not steps
    assert report["status"] == "webcam measurement unavailable"


def test_stop_prevents_movement_and_observation():
    report = refine(lambda: 1 / 0, lambda delta: 1 / 0, lambda: True)
    assert report["status"] == "operator stopped"


def test_budget_limits_unresponsive_or_badly_scaled_control():
    calls = []
    report = refine(
        lambda: dict(error_px=[100, 100], jacobian_px_m=np.eye(2) * 100),
        lambda delta: calls.append(delta) or np.zeros(2),
        lambda: False,
    )
    assert len(calls) == 1
    assert report["status"] == "no measured response"


def test_webcam_snapshot_does_not_require_realsense_file(tmp_path):
    import os
    import time

    from PIL import Image

    from deskbuddy.webcam_servo import read_frame

    path = tmp_path / "webcam.jpg"
    Image.new("RGB", (20, 20)).save(path)
    stamp = time.time()
    os.utime(path, (stamp, stamp))
    result = read_frame(
        lambda: dict(
            camera_stale=True,
            pickup_rgb_unix_s=stamp,
            pickup_rgb_paths=["missing-realsense.jpg", str(path)],
        ),
        0,
    )
    assert result[1].shape == (20, 20, 3)


def test_webcam_error_compensates_hover_height_instead_of_aligning_to_table_pixel(
    monkeypatch,
):
    from deskbuddy import can_discard, webcam_servo

    monkeypatch.setattr(
        webcam_servo, "corners", lambda *args, **kwargs: np.zeros((4, 3))
    )

    def projection(points):
        points = np.asarray(points)
        return points[:, :2] * 1000 + points[:, 2, None] * np.array([100, 200])

    monkeypatch.setattr(webcam_servo, "projector", lambda *args: projection)
    monkeypatch.setattr(webcam_servo, "finger_points", lambda *args: np.zeros((4, 3)))
    monkeypatch.setattr(webcam_servo, "outline", lambda points: points)
    monkeypatch.setattr(
        can_discard, "jaw_center", lambda *args: np.array([0.2, 0.1, 0.2])
    )
    fits = iter(
        [dict(chamfer_px=0.5, shift_px=[4, -3]), dict(chamfer_px=0.5, shift_px=[1, 2])]
    )
    monkeypatch.setattr(webcam_servo, "fit_lid", lambda *args, **kwargs: next(fits))
    config = dict(
        secondary_camera_calibration=dict(
            T_base_camera=np.eye(4).tolist(), intrinsics={}, secondary_pixels=[]
        )
    )
    result = webcam_servo.measurement(
        np.zeros((200, 300, 3), np.uint8),
        config,
        None,
        None,
        30,
        dict(center_m=[0.2, 0.1, 0], height_m=0.122, diameter_m=0.066),
        [0.2, 0.1, 0.05],
    )
    np.testing.assert_allclose(result["error_px"], [3, -5])
    np.testing.assert_allclose(result["jacobian_px_m"], np.eye(2) * 1000, atol=0.001)


def test_probes_measure_jacobian_instead_of_trusting_bad_model():
    from deskbuddy.local_visual_servo import measure_jacobian

    actual = np.array([[1000.0, 250.0], [-200.0, 800.0]])
    position = np.zeros(2)
    steps = []

    def observe():
        return dict(error_px=actual @ position + [5.0, -3.0], jacobian_px_m=np.eye(2))

    def move(delta):
        steps.append(delta.copy())
        position[:] += delta * 0.8
        return delta * 0.8

    measured = measure_jacobian(observe, move, lambda: False)
    assert measured["ready"]
    np.testing.assert_allclose(measured["jacobian_px_m"], actual, atol=1e-7)
    assert len(steps) == 4
    assert all(np.linalg.norm(step) <= 0.00200001 for step in steps)
    # Return callback is commanded using achieved displacement; its own
    # response can lag, so the residual model position is reported by sensing.
    assert np.linalg.norm(position) < 0.0005


def test_missing_probe_image_returns_before_accepting_a_jacobian():
    from deskbuddy.local_visual_servo import measure_jacobian

    observations = iter([dict(error_px=[0, 0]), None, dict(error_px=[0, 0])])
    moves = []
    result = measure_jacobian(
        lambda: next(observations), lambda d: moves.append(d.copy()) or d, lambda: False
    )
    assert not result["ready"]
    assert len(moves) == 2
    np.testing.assert_allclose(moves[0] + moves[1], 0)


def test_selected_fixed_tip_uses_measured_pixel_without_edge_or_color_detection(
    monkeypatch,
):
    from deskbuddy import can_discard, webcam_servo

    monkeypatch.setattr(webcam_servo, "corners", lambda *a, **k: np.zeros((4, 3)))

    def projection(points):
        points = np.asarray(points)
        return points[:, :2] * 1000 + points[:, 2, None] * np.array([100, 200])

    monkeypatch.setattr(webcam_servo, "projector", lambda *a: projection)
    monkeypatch.setattr(webcam_servo, "finger_points", lambda *a: 1 / 0)
    monkeypatch.setattr(can_discard, "jaw_center", lambda *a: np.array([0.2, 0.1, 0.2]))
    monkeypatch.setattr(
        can_discard,
        "fingertips",
        lambda *a: (np.array([0.19, 0.1, 0.2]), np.array([0.21, 0.1, 0.2])),
    )
    monkeypatch.setattr(
        webcam_servo, "fit_lid", lambda *a, **k: dict(chamfer_px=0.5, shift_px=[1, 2])
    )
    config = dict(
        secondary_camera_calibration=dict(
            T_base_camera=np.eye(4).tolist(), intrinsics={}, secondary_pixels=[]
        )
    )
    result = webcam_servo.measurement(
        np.zeros((200, 300, 3), np.uint8),
        config,
        None,
        None,
        30,
        dict(center_m=[0.2, 0.1, 0], height_m=0.122, diameter_m=0.066),
        [0.2, 0.1, 0.05],
        tracked_tip=[214, 137],
    )
    np.testing.assert_allclose(result["error_px"], [3, -5])
