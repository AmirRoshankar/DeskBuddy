"""Conditional interior-waypoint models; physical endpoints are enforced exactly."""

import numpy as np
import torch
from torch import nn


def condition(case):
    scene = case["scene"]
    low, high = np.asarray(scene["low"]), np.asarray(scene["high"])
    scale = high - low
    result = list((np.asarray(case["start"]) - low) / scale) + list(
        (np.asarray(case["goal"]) - low) / scale
    )
    obstacles = []
    for cx, cy, radius in scene["circles"]:
        obstacles.append(
            [1, 0, *((np.array([cx, cy]) - low) / scale), *(radius / scale)]
        )
    for box in scene["boxes"]:
        a, b = np.asarray(box[:2]), np.asarray(box[2:])
        obstacles.append(
            [1, 1, *((((a + b) / 2) - low) / scale), *((b - a) / 2 / scale)]
        )
    if len(obstacles) > 2:
        raise ValueError("This experiment supports at most two obstacles")
    result += [
        value for row in obstacles + [[0] * 6] * (2 - len(obstacles)) for value in row
    ]
    result += list((scene["tool_radius"] + scene["margin"]) / scale)
    return np.asarray(result, np.float32)


def encode_path(path, case):
    scale = np.asarray(case["scene"]["high"]) - case["scene"]["low"]
    line = np.linspace(case["start"], case["goal"], 32)
    return ((np.asarray(path)[1:-1] - line[1:-1]) / scale).reshape(60)


def decode_path(residual, case):
    scale = np.asarray(case["scene"]["high"]) - case["scene"]["low"]
    path = np.linspace(case["start"], case["goal"], 32)
    path[1:-1] += np.asarray(residual).reshape(30, 2) * scale
    return path


class Dense(nn.Module):
    def __init__(self, inputs, outputs=60, width=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, outputs),
        )

    def forward(self, value):
        return self.net(value)


class Field(nn.Module):
    def __init__(self, width=192, auxiliary=False, condition_dim=18, state_dim=60):
        super().__init__()
        self.net = Dense(condition_dim + state_dim + 2, outputs=state_dim, width=width)
        self.velocity_net = (
            Dense(condition_dim + state_dim + 1, outputs=state_dim, width=width)
            if auxiliary
            else None
        )

    def forward(self, z, r, t, c):
        return z + self.net(torch.cat((c, z, r, t), dim=-1))

    def instantaneous(self, z, t, c):
        if self.velocity_net is None:
            return self(z, t, t, c)
        return z + self.velocity_net(torch.cat((c, z, t), dim=-1))


def loss(
    model, c, data, method, warmup=False, diagonal_fraction=0.75, adaptive_power=1.0
):
    if method == "mlp":
        return (model(c) - data).square().mean()
    t, r = torch.rand((2, len(data), 1), device=data.device).sort(dim=0).values.flip(0)
    r = torch.where(torch.rand_like(t) < (1.0 if warmup else diagonal_fraction), t, r)
    noise = torch.randn_like(data)
    z = (1 - t) * data + t * noise
    velocity = noise - data
    if method == "fm":
        return (model(z, t, t, c) - velocity).square().mean()

    def fn(z, r, t):
        return model(z, r, t, c)

    if method == "imf":
        with torch.no_grad():
            tangent = model.instantaneous(z, t, c)
    elif method == "mf":
        tangent = velocity
    else:
        raise ValueError("Unknown trajectory method")
    average, derivative = torch.func.jvp(
        fn, (z, r, t), (tangent, torch.zeros_like(r), torch.ones_like(t))
    )
    if method == "mf":
        error = average - (velocity - (t - r) * derivative).detach()
    else:
        error = average + (t - r) * derivative.detach() - velocity
    squared = error.square().mean(dim=1)
    weight = (squared.detach() + 1e-3).pow(-adaptive_power)
    result = (weight * squared).mean()
    if method == "imf" and model.velocity_net is not None:
        auxiliary = (model.instantaneous(z, t, c) - velocity).square().mean(dim=1)
        result = (
            result
            + (auxiliary / (auxiliary.detach() + 1e-3).pow(adaptive_power)).mean()
        )
    return result


def sample(model, c, noise, method, steps=1):
    if steps < 1:
        raise ValueError("Need a positive integration step count")
    if method == "mlp":
        return model(c)
    z = noise.clone()
    for index in range(steps):
        t = z.new_full((len(z), 1), 1 - index / steps)
        r = z.new_full((len(z), 1), 1 - (index + 1) / steps)
        z = z - model(z, t if method == "fm" else r, t, c) / steps
    return z
