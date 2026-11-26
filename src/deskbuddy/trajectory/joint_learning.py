"""Differentiable whole-arm geometry loss for a future 5D trajectory training run.

The current XY checkpoints do not use this loss. Robot FK is the same URDF chain
as runtime; sampled sphere/box costs guide training but do not certify paths.
"""

import torch
from torch import nn


class TorchArm(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.active = []
        for i, (_, origin, axis, bounds) in enumerate(arm.chain):
            self.register_buffer(
                f"origin_{i}", torch.tensor(origin, dtype=torch.float32)
            )
            self.register_buffer(f"axis_{i}", torch.tensor(axis, dtype=torch.float32))
            self.active.append(bounds is not None)
        self.register_buffer("tool", torch.tensor(arm.tool_offset, dtype=torch.float32))
        self.has_tool = bool((arm.tool_offset != 0).any())

    def forward(self, q):
        transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(
            *q.shape[:-1], 4, 4
        )
        points = [transform[..., :3, 3]]
        index = 0
        for i, active in enumerate(self.active):
            transform = transform @ getattr(self, f"origin_{i}")
            if active:
                axis = getattr(self, f"axis_{i}")
                zero = axis.new_zeros(())
                cross = torch.stack(
                    (
                        torch.stack((zero, -axis[2], axis[1])),
                        torch.stack((axis[2], zero, -axis[0])),
                        torch.stack((-axis[1], axis[0], zero)),
                    )
                )
                angle = q[..., index, None, None]
                rotation = (
                    torch.eye(3, device=q.device, dtype=q.dtype)
                    + torch.sin(angle) * cross
                    + (1 - torch.cos(angle)) * (cross @ cross)
                )
                motion = (
                    torch.eye(4, device=q.device, dtype=q.dtype)
                    .expand(*q.shape[:-1], 4, 4)
                    .clone()
                )
                motion[..., :3, :3] = rotation
                transform = transform @ motion
                index += 1
            points.append(transform[..., :3, 3])
        if self.has_tool:
            points.append(
                transform[..., :3, 3]
                + (transform[..., :3, :3] @ self.tool[..., None]).squeeze(-1)
            )
        return torch.stack(points, dim=-2)


def whole_arm_collision_loss(fk, q, boxes, present):
    """q: B,T,5; boxes: B,K,2,3; present: B,K. All units metres/radians."""
    points = fk(q)
    weights = torch.linspace(0, 1, 7, device=q.device, dtype=q.dtype)
    samples = (
        points[..., :-1, None, :] * (1 - weights[:, None])
        + points[..., 1:, None, :] * weights[:, None]
    )
    radii = q.new_full((points.shape[-2] - 1,), 0.035)
    radii[-1] = 0.045
    center = (boxes[:, :, 0] + boxes[:, :, 1]) / 2
    half = (boxes[:, :, 1] - boxes[:, :, 0]) / 2
    delta = (samples.unsqueeze(-2) - center[:, None, None, None, :, :]).abs() - half[
        :, None, None, None, :, :
    ]
    signed = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(
        dim=-1
    ).clamp_max(0)
    penalty = (
        torch.relu(radii[None, None, :, None, None] - signed).square()
        * present[:, None, None, None, :]
    )
    total = penalty.mean()
    pairs = []
    for i in range(samples.shape[-3]):
        for j in range(i + 3, samples.shape[-3]):
            if i >= 4 and j >= 4:
                continue
            distance = torch.cdist(samples[..., i, :, :], samples[..., j, :, :]).amin(
                dim=(-1, -2)
            )
            pairs.append(torch.relu(0.04 - distance).square().mean())
    if pairs:
        total = total + torch.stack(pairs).mean()
    return total
