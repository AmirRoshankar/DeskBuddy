"""CPU IK initial guesses; current geometry is enforced by refinement, not training."""

from pathlib import Path

import numpy as np
import torch

from .data import signature
from .models import MeanFlow, Network


class Proposal:
    def __init__(self, path, config, method="imf"):
        torch.set_num_threads(1)
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != 1 or checkpoint.get("method") != method:
            raise ValueError("Unsupported learned IK checkpoint")
        # A changed selection box or calibration does not invalidate an initial
        # guess. Keep this provenance visible, while refining and checking every
        # proposal against the current Arm/config before any execution.
        self.configuration_matches = checkpoint["signature"] == signature(config)
        self.training_config = checkpoint.get("config")
        self.method = method
        self.normalization = checkpoint["normalization"]
        self.model = (
            MeanFlow(checkpoint["width"])
            if method == "imf"
            else Network(8, checkpoint["width"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.noise = torch.randn((1, 4), generator=torch.Generator().manual_seed(42))

    def in_training_domain(self, start, target):
        x = np.r_[start, target]
        n = self.normalization
        return bool(
            (
                (x >= np.array(n["x_min"]) - 0.01) & (x <= np.array(n["x_max"]) + 0.01)
            ).all()
        )

    def predict(self, start, target, check_domain=False):
        if (
            np.shape(start) != (5,)
            or np.shape(target) != (3,)
            or not np.isfinite(np.r_[start, target]).all()
        ):
            raise ValueError(
                "Expected five finite joint angles and a finite XYZ target"
            )
        x = np.r_[start, target]
        n = self.normalization
        # Strict-domain evaluation remains available for experiments. Deployment
        # permits extrapolated guesses, reports them, and validates after IK.
        if check_domain and not self.in_training_domain(start, target):
            raise ValueError(
                "Selected state/target is outside learned training coverage"
            )
        condition = torch.tensor(
            (x - n["x_mean"]) / n["x_std"], dtype=torch.float32
        ).reshape(1, -1)
        with torch.inference_mode():
            result = (
                self.model.sample(condition, self.noise)
                if self.method == "imf"
                else self.model(condition)
            )
        delta = result.numpy()[0] * n["y_std"] + n["y_mean"]
        proposed = np.asarray(start, float).copy()
        proposed[:4] += delta
        if not np.isfinite(proposed).all():
            raise ValueError("Learned proposal returned nonfinite angles")
        return proposed


_CACHE = {}


def get_proposal(path, config, method="imf"):
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, signature(config), method)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = Proposal(path, config, method)
    return _CACHE[key]
