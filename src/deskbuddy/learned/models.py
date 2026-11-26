"""Small state-conditioned MLPs and the iMF v-loss (Geng et al., Algorithm 1)."""

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, inputs, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 4),
        )

    def forward(self, x):
        return self.net(x)


class MeanFlow(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.field = Network(14, width)

    def forward(self, z, r, t, condition):
        return z + self.field(torch.cat((condition, z, r, t), dim=-1))

    def sample(self, condition, noise):
        zero = condition.new_zeros((len(condition), 1))
        return noise - self(noise, zero, zero + 1, condition)


def imf_loss(model, condition, data, warmup=False):
    """Predicted marginal-velocity tangent; stop-gradient on the JVP outcome.

    This is Algorithm 1 of arXiv:2512.02012, without CFG or an image backbone.
    r=t examples anchor ordinary flow matching. Flow time is not robot time.
    """
    t, r = torch.rand((2, len(data), 1), device=data.device).sort(dim=0).values.flip(0)
    diagonal = torch.rand_like(t) < (1.0 if warmup else 0.25)
    r = torch.where(diagonal, t, r)
    noise = torch.randn_like(data)
    z = (1 - t) * data + t * noise

    def fn(z, r, t):
        return model(z, r, t, condition)

    with torch.no_grad():
        velocity = fn(z, t, t)
    average, derivative = torch.func.jvp(
        fn, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t))
    )
    predicted = average + (t - r) * derivative.detach()
    return (predicted - (noise - data)).square().mean()
