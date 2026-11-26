import numpy as np
import pytest

from deskbuddy.kinematics import Arm
from deskbuddy.trajectory.joint_learning import TorchArm, whole_arm_collision_loss
from deskbuddy.trajectory.learning import Field, loss

torch = pytest.importorskip("torch")


def test_training_fk_matches_runtime_and_collision_loss_has_joint_gradients():
    arm = Arm(tool_offset_m=[0.005, 0, 0])
    q = torch.tensor([[[0.1, -0.3, 0.5, 0.3, 0.05]]], requires_grad=True)
    fk = TorchArm(arm)
    points = fk(q)
    expected = arm.frames(q.detach().numpy()[0, 0])[0]
    assert np.allclose(points.detach().numpy()[0, 0], expected, atol=1e-6)
    center = points.detach()[0, 0, 4] + torch.tensor([0.025, 0.01, 0])
    boxes = torch.stack((center - 0.01, center + 0.01)).reshape(1, 1, 2, 3)
    value = whole_arm_collision_loss(fk, q, boxes, torch.ones(1, 1))
    value.backward()
    assert value > 0 and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_meanflow_field_supports_full_joint_trajectories_without_changing_old_shapes():
    model = Field(width=16, auxiliary=True, condition_dim=24, state_dim=150)
    value = loss(model, torch.randn(3, 24), torch.randn(3, 150), "imf")
    assert torch.isfinite(value)
    value.backward()
    assert Field(width=16)(
        torch.zeros(2, 60), torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(2, 18)
    ).shape == (2, 60)
