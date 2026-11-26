"""Check the implemented sign/time convention against an analytic Gaussian flow."""

import pytest

torch = pytest.importorskip("torch")


def test_imf_velocity_identity_jvp_and_one_step_sampling_match_analytic_flow():
    from deskbuddy.trajectory.learning import sample

    class GaussianField(torch.nn.Module):
        def sigma(self, t):
            return (0.35**2 * (1 - t) ** 2 + t * t).sqrt()

        def velocity(self, z, t, c):
            slope = (t - 0.35**2 * (1 - t)) / self.sigma(t) ** 2
            return slope * (z - (1 - t) * c) - c

        def forward(self, z, r, t, c):
            mapped = self.sigma(r) / self.sigma(t) * (z - (1 - t) * c) + (1 - r) * c
            return (z - mapped) / (t - r)

    model = GaussianField()
    torch.manual_seed(15)
    c = torch.randn(16, 3, dtype=torch.float64)
    z = torch.randn_like(c)
    r = torch.full((16, 1), 0.2, dtype=torch.float64)
    t = torch.full((16, 1), 0.7, dtype=torch.float64)
    v = model.velocity(z, t, c)

    def fn(z, r, t):
        return model(z, r, t, c)

    u, derivative = torch.func.jvp(
        fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
    )
    assert torch.allclose(u + (t - r) * derivative, v, atol=1e-10)
    epsilon = 1e-5
    numeric = (
        fn(z + epsilon * v, r, t + epsilon) - fn(z - epsilon * v, r, t - epsilon)
    ) / (2 * epsilon)
    assert torch.allclose(derivative, numeric, atol=1e-8)
    noise = torch.randn_like(c)
    predicted = sample(model, c, noise, "imf", 1)
    assert torch.allclose(predicted, c + 0.35 * noise, atol=1e-10)
