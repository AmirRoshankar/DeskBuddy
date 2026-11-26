import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deskbuddy.learned.adaptive_grasp import (  # noqa: E402
    ActionField,
    objective,
    replan,
    window_data,
)
from deskbuddy.learned.grasp_lab import (  # noqa: E402
    GraspMF,
    advance,
    box_data,
    delta,
    geometric_metrics,
    pose_noise,
    rotation_exp,
    rotation_log,
    unit,
)


def test_group_interpolation_preserves_rotations_and_endpoint_and_has_finite_gradients():
    torch.manual_seed(4)
    a, b = pose_noise(16, "cpu"), pose_noise(16, "cpu")
    v = delta(a, b)
    end = advance(a, v, torch.ones(16, 1))
    assert torch.allclose(end[:, :3], b[:, :3], atol=1e-6)
    assert torch.allclose(
        abs((end[:, 3:7] * b[:, 3:7]).sum(-1)), torch.ones(16), atol=1e-6
    )
    assert torch.allclose(end[:, 3:7].norm(dim=-1), torch.ones(16), atol=1e-6)
    vectors = torch.tensor([[0.2, -0.3, 0.4]], requires_grad=True)
    assert torch.allclose(rotation_log(rotation_exp(vectors)), vectors, atol=1e-6)
    rotation_log(rotation_exp(vectors)).square().sum().backward()
    assert torch.isfinite(vectors.grad).all()


def test_procedural_grasp_labels_pass_only_the_declared_geometry_metric():
    c, y, ids = box_data(40)
    assert (
        geometric_metrics(torch.tensor(y), torch.tensor(c))["geometric_valid_rate"]
        == 1.0
    )
    assert np.array_equal(ids[::2], ids[1::2])
    assert not np.array_equal(
        y[0, 3:7], y[1, 3:7]
    )  # two distinct face-aligned grasp modes
    model = GraspMF(width=32)
    value = model.objective(torch.tensor(y[:8]), torch.tensor(c[:8]))
    value.backward()
    assert torch.isfinite(value)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("method", ["adaptive_mf", "adaptive_imf"])
def test_vector_time_action_training_and_shifted_prior_replanning_are_finite(method):
    c, y, _ = window_data(20)
    c, y = torch.tensor(c[:8]), torch.tensor(y[:8])
    model = ActionField(width=32)
    value = objective(model, y, c, method)
    value.backward()
    assert torch.isfinite(value)
    with torch.no_grad():
        first = replan(model, c)
        second = replan(model, c, first)
    assert first.shape == second.shape == (8, 16, 6)
    assert torch.isfinite(second).all()


def test_unit_quaternion_handles_zero_and_sign_equivalence():
    assert torch.equal(unit(torch.zeros(1, 4)), torch.tensor([[1.0, 0, 0, 0]]))
    q = unit(torch.randn(4, 4))
    assert torch.allclose(rotation_log(q), rotation_log(-q), atol=1e-6)


def test_cylinder_teacher_allows_rotational_modes_and_heldout_object_groups():
    from deskbuddy.learned.grasp_lab import can_data, geometric_metrics

    c, y, groups = can_data(20)
    assert len(c) == 80 and len(np.unique(groups)) == 20
    assert geometric_metrics(
        torch.tensor(y), torch.tensor(c), True, profile="can"
    ).all()
