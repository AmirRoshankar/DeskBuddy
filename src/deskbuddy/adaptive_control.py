"""Bounded outer PI correction around existing position-mode servo controllers."""

import numpy as np

# Software envelope, not a measured STS3215 hardware specification.
HARDWARE_SPEED_LIMIT_DEG_S = 30.0


class AdaptiveCorrection:
    def __init__(
        self,
        bias,
        initial_command,
        lower,
        upper,
        max_speed_deg_s=12.0,
        locked=None,
        kp=0.2,
        ki=1.0,
    ):
        self.integral = -np.asarray(bias, float).copy()
        self.initial = np.asarray(initial_command, float).copy()
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.bound = np.maximum(np.deg2rad(12.0), np.abs(self.integral))
        self.max_speed = np.deg2rad(max_speed_deg_s)
        self.locked = (
            np.zeros(5, dtype=bool) if locked is None else np.asarray(locked, bool)
        )
        self.kp = kp
        self.ki = ki

    def propose(self, reference, estimate, previous, dt, freeze_integral=False):
        error = np.asarray(reference) - estimate
        integral = (
            self.integral.copy()
            if freeze_integral
            else np.clip(self.integral + self.ki * error * dt, -self.bound, self.bound)
        )
        raw = np.asarray(reference) + integral + self.kp * error
        step = min(self.max_speed * dt, np.deg2rad(0.4))
        command = np.clip(
            np.clip(raw, np.asarray(previous) - step, np.asarray(previous) + step),
            self.lower,
            self.upper,
        )
        # Conditional anti-windup: do not integrate farther into a saturated goal.
        saturated = error * (raw - command) > 1e-12
        integral[saturated] = self.integral[saturated]
        integral[self.locked] = self.integral[self.locked]
        command[self.locked] = self.initial[self.locked]
        return command, integral, error

    def commit(self, integral):
        self.integral = integral.copy()
