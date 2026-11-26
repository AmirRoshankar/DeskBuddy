from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.hardware_interlock import require_hardware_enabled
from deskbuddy.kinematics import Arm
from deskbuddy.learned.data import signature


def test_hardware_pause_blocks_actions_before_process_creation(tmp_path, monkeypatch):
    from deskbuddy.go_control import launch, launch_calibration_action

    monkeypatch.chdir(tmp_path)
    Path("runs/control").mkdir(parents=True)
    Path("runs/control/hardware_disabled").write_text("operator instruction")
    with pytest.raises(RuntimeError, match="Explicit resumption"):
        require_hardware_enabled()
    with pytest.raises(RuntimeError, match="Hardware disabled"):
        launch(
            None,
            None,
            None,
            popen=lambda *args, **kwargs: pytest.fail("No process allowed"),
        )
    with pytest.raises(RuntimeError, match="Hardware disabled"):
        launch_calibration_action("hold", {})


def test_signature_invalidates_changed_geometry_but_not_display_or_speed():
    config = deepcopy(DEFAULT)
    original = signature(config)
    config["hover_m"] = 0.15
    config["max_joint_speed_deg_s"] = 12
    assert signature(config) == original
    config["tool_offset_m"][0] = 0.005
    assert signature(config) != original


def test_artifact_budget_is_checked_before_allocating_more(tmp_path, monkeypatch):
    from deskbuddy.learned import data

    monkeypatch.setattr(data, "MAX_BYTES", 100)
    (tmp_path / "existing.bin").write_bytes(bytes(80))
    assert data.check_budget(tmp_path, 20) == 80
    with pytest.raises(RuntimeError, match="10 GB"):
        data.check_budget(tmp_path, 21)


def test_nested_experiments_share_one_folder_budget(tmp_path, monkeypatch):
    from deskbuddy.learned import data

    monkeypatch.setattr(data, "MAX_BYTES", 100)
    (tmp_path / ".resource-budget.json").write_text("{}")
    (tmp_path / "base.bin").write_bytes(bytes(80))
    nested = tmp_path / "multimodal"
    nested.mkdir()
    with pytest.raises(RuntimeError, match="10 GB"):
        data.check_budget(nested, 19)


def test_distribution_score_rewards_both_modes_instead_of_their_mean():
    torch = pytest.importorskip("torch")
    from deskbuddy.learned.train import energy_score

    observed = torch.tensor([[-1.0], [1.0]])
    samples = torch.tensor([-1.0, 1.0] * 32).reshape(1, 64, 1).repeat(2, 1, 1)
    distribution = energy_score(samples, observed)
    mean_prediction = energy_score(torch.zeros((2, 1, 1)), observed)
    assert distribution < 0.6 and mean_prediction == 1.0


def test_virtual_field_trace_preserves_locked_joint_and_time_direction():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from deskbuddy.learned.visualize import field_trace

    wrapper = SimpleNamespace(
        normalization=dict(
            x_mean=np.zeros(8), x_std=np.ones(8), y_mean=np.zeros(4), y_std=np.ones(4)
        ),
        model=lambda z, r, t, c: torch.ones_like(z),
    )
    x = np.zeros(8)
    x[4] = 0.3
    progress, states, velocities = field_trace(wrapper, x, torch.zeros((1, 4)), steps=5)
    assert np.allclose(states[-1, :4], -1.0)
    assert np.all(states[:, 4] == 0.3)
    assert np.allclose(velocities, -180 / np.pi)
    assert progress[0] == 0 and progress[-1] == 1


def test_warm_start_preserves_posture_reference_and_locked_wrist():
    arm = Arm()
    start = np.deg2rad(DEFAULT["preview_start_deg"])
    goal = start + np.deg2rad([4, 2, -3, 2, 0])
    guess = goal.copy()
    guess[4] += 0.3
    solution = arm.ik(
        arm.fk(goal), start, locked_joints=["wrist_roll"], initial_guess=guess
    )
    assert solution[4] == start[4]
    assert np.linalg.norm(arm.fk(solution) - arm.fk(goal)) < 0.002


def test_flow_refines_good_proposal_and_falls_back_for_bad_one(monkeypatch):
    pytest.importorskip("torch")
    from deskbuddy.learned import inference
    from deskbuddy.learned.solver import solve_goal

    arm = Arm()
    config = deepcopy(DEFAULT)
    config["planner_mode"] = "flow_assisted"
    start = np.deg2rad(config["preview_start_deg"])
    expected = start + np.deg2rad([4, 2, -3, 2, 0])
    target = arm.fk(expected)

    class Proposal:
        def predict(self, *args):
            return expected + np.deg2rad([0.2, 0.1, -0.2, 0.1, 0])

    monkeypatch.setattr(inference, "get_proposal", lambda *args, **kwargs: Proposal())
    solution, info = solve_goal(arm, start, target, config)
    assert info["used"] == "flow_assisted"
    assert np.linalg.norm(arm.fk(solution) - target) < 0.002
    monkeypatch.setattr(Proposal, "predict", lambda *args: np.full(5, np.nan))
    solution, info = solve_goal(arm, start, target, config)
    assert info["used"] == "numerical" and "finite" in info["fallback_reason"]
    assert np.linalg.norm(arm.fk(solution) - target) < 0.002


def test_small_learned_overshoot_is_a_refinable_seed_not_an_invalid_goal(monkeypatch):
    pytest.importorskip("torch")
    from deskbuddy.learned import inference
    from deskbuddy.learned.solver import solve_goal

    arm = Arm()
    config = dict(DEFAULT, planner_mode="flow_assisted")
    start = np.deg2rad(config["preview_start_deg"])
    # A valid target near the elbow boundary, with an imperfect learned guess.
    expected = start.copy()
    expected[2] = arm.upper[2] - np.deg2rad(0.3)
    predicted = expected.copy()
    predicted[2] = arm.upper[2] + np.deg2rad(0.15)

    class Proposal:
        def predict(self, *args):
            return predicted.copy()

    monkeypatch.setattr(inference, "get_proposal", lambda *args, **kwargs: Proposal())
    target = arm.fk(expected)
    goal, info = solve_goal(arm, start, target, config)
    assert info["used"] == "flow_assisted"
    assert info["seed_projection_deg"][2] == pytest.approx(-0.15)
    assert (goal >= arm.lower).all() and (goal <= arm.upper).all()
    assert np.linalg.norm(arm.fk(goal) - target) < config["ik_tolerance_m"]
    # Projection must not quietly change the model output stored by inference.
    assert predicted[2] > arm.upper[2]


def test_imf_loss_has_finite_gradients_and_one_step_keeps_batch_shape():
    torch = pytest.importorskip("torch")
    from deskbuddy.learned.models import MeanFlow, imf_loss

    torch.manual_seed(2)
    model = MeanFlow(32)
    condition, data = torch.randn(8, 8), torch.randn(8, 4)
    loss = imf_loss(model, condition, data)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert model.sample(condition, torch.randn_like(data)).shape == data.shape


@pytest.mark.parametrize("method", ["imf", "mlp"])
def test_changed_bounds_and_tool_calibration_still_use_learned_seed(tmp_path, method):
    torch = pytest.importorskip("torch")
    from deskbuddy.learned.models import MeanFlow, Network
    from deskbuddy.learned.solver import solve_goal

    training = deepcopy(DEFAULT)
    model = MeanFlow(16) if method == "imf" else Network(8, 16)
    for parameter in model.parameters():
        parameter.data.zero_()
    path = tmp_path / (method + ".pt")
    torch.save(
        dict(
            schema=1,
            method=method,
            width=16,
            signature=signature(training),
            state_dict=model.state_dict(),
            normalization=dict(
                x_mean=[0.0] * 8,
                x_std=[1.0] * 8,
                x_min=[0.0] * 8,
                x_max=[0.0] * 8,
                y_mean=[0.0] * 4,
                y_std=[0.001] * 4,
            ),
        ),
        path,
    )
    current = dict(
        training,
        workspace_min_m=[0.01, -0.5, 0.01],
        workspace_max_m=[0.8, 0.5, 0.6],
        tool_offset_m=[0.004, 0, 0],
        planner_mode="flow_assisted",
        proposal_method=method,
        flow_checkpoint=str(path),
    )
    arm = Arm(tool_offset_m=current["tool_offset_m"])
    start = np.deg2rad(current["preview_start_deg"])
    target = arm.fk(start + np.deg2rad([2, 3, -2, 1, 0]))
    goal, info = solve_goal(arm, start, target, current)
    assert info["used"] == "flow_assisted"
    assert info["training_config_changed"] and info["outside_training_coverage"]
    assert np.linalg.norm(arm.fk(goal) - target) < current["ik_tolerance_m"]
    assert (goal >= arm.lower).all() and (goal <= arm.upper).all()


def test_fallback_log_explicitly_reports_requested_and_actual_solver():
    from deskbuddy.learned.solver import planner_status

    diagnostics = [
        dict(requested="flow_assisted", used="numerical", fallback_reason="IK failed")
    ]
    message = planner_status(diagnostics, "flow_assisted")
    assert "Numerical fallback" in message
    assert "IK failed" in message
    diagnostics[0].update(used="flow_assisted", proposal_method="imf")
    assert planner_status(diagnostics) == ""
    assert planner_status([dict(requested="numerical", used="numerical")]) == ""


def test_affine_seed_is_refined_to_original_physical_target(monkeypatch):
    pytest.importorskip("torch")
    from deskbuddy.learned import inference
    from deskbuddy.learned.coordinates import map_workspace_target
    from deskbuddy.learned.solver import solve_goal

    config = dict(
        deepcopy(DEFAULT),
        planner_mode="flow_assisted",
        flow_seed_strategy="workspace_affine",
    )
    training = dict(
        config, workspace_min_m=[0.1, -0.2, 0.05], workspace_max_m=[0.3, 0.2, 0.3]
    )
    arm = Arm()
    start = np.deg2rad(config["preview_start_deg"])
    target = arm.fk(start + np.deg2rad([2, 3, -2, 1, 0]))
    calls = []

    class Model:
        training_config = training

        def predict(self, q, point):
            calls.append(np.array(point))
            return start.copy()

    monkeypatch.setattr(inference, "get_proposal", lambda *a, **k: Model())
    goal, info = solve_goal(arm, start, target, config)
    assert (
        info["used"] == "flow_assisted" and info["seed_strategy"] == "workspace_affine"
    )
    assert np.allclose(calls[0], map_workspace_target(target, config, training))
    assert not np.allclose(calls[0], target)
    assert np.linalg.norm(arm.fk(goal) - target) < config["ik_tolerance_m"]
