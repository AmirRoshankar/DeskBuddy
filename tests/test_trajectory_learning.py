import numpy as np
import pytest

from deskbuddy.trajectory.dataset import layout


@pytest.mark.parametrize("method", ["mlp", "fm", "mf", "imf", "imf_aux"])
def test_trajectory_losses_have_finite_gradients_and_samples(method):
    torch = pytest.importorskip("torch")
    from deskbuddy.trajectory.learning import Dense, Field, loss, sample

    torch.manual_seed(42)
    auxiliary = method == "imf_aux"
    if auxiliary:
        method = "imf"
    model = Dense(18, width=16) if method == "mlp" else Field(16, auxiliary=auxiliary)
    c, data = torch.randn(8, 18), torch.randn(8, 60)
    value = loss(model, c, data, method)
    value.backward()
    assert torch.isfinite(value)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert sample(model, c, torch.randn_like(data), method, 4).shape == (8, 60)


def test_cartesian_encoding_roundtrip_preserves_original_endpoints():
    pytest.importorskip("torch")
    from dataclasses import asdict

    from deskbuddy.trajectory.learning import condition, decode_path, encode_path

    case = dict(scene=asdict(layout(2, 42)), start=[0.02, 0.02], goal=[0.19, 0.25])
    path = np.linspace(case["start"], case["goal"], 32)
    path[1:-1, 1] += np.sin(np.linspace(0, np.pi, 30)) * 0.03
    context = condition(case)
    assert context.shape == (18,) and np.isfinite(context).all()
    assert np.allclose(decode_path(encode_path(path, case), case), path)
    changed = decode_path(np.ones(60), case)
    assert np.array_equal(changed[0], case["start"]) and np.array_equal(
        changed[-1], case["goal"]
    )
