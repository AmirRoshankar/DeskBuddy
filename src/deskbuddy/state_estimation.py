"""Independent constant-velocity Kalman states for the measured arm joints."""

import numpy as np


class JointKalman:
    def __init__(self, position, encoder_std_deg=0.05, acceleration_std_deg_s2=40.0):
        self.position = np.asarray(position, float).copy()
        self.velocity = np.zeros_like(self.position)
        self.measurement_variance = np.deg2rad(encoder_std_deg) ** 2
        self.acceleration_variance = np.deg2rad(acceleration_std_deg_s2) ** 2
        self.p00 = np.full_like(self.position, self.measurement_variance)
        self.p01 = np.zeros_like(self.position)
        self.p11 = np.full_like(self.position, np.deg2rad(5.0) ** 2)

    def predict(self, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Invalid filter time interval")
        self.position += self.velocity * dt
        q = self.acceleration_variance
        self.p00 += 2 * dt * self.p01 + dt**2 * self.p11 + q * dt**4 / 4
        self.p01 += dt * self.p11 + q * dt**3 / 2
        self.p11 += q * dt**2

    def update(self, measurement):
        measurement = np.asarray(measurement, float)
        if (
            measurement.shape != self.position.shape
            or not np.isfinite(measurement).all()
        ):
            raise ValueError("Invalid Kalman measurement")
        innovation = measurement - self.position
        r = self.measurement_variance
        k0 = self.p00 / (self.p00 + r)
        k1 = self.p01 / (self.p00 + r)
        self.position += k0 * innovation
        self.velocity += k1 * innovation
        p00, p01, p11 = self.p00.copy(), self.p01.copy(), self.p11.copy()
        # Joseph covariance update, preserving numerical positivity.
        self.p00 = (1 - k0) ** 2 * p00 + k0**2 * r
        self.p01 = (1 - k0) * (p01 - k1 * p00) + k0 * k1 * r
        self.p11 = p11 - 2 * k1 * p01 + k1**2 * (p00 + r)
        return self.position.copy()

    @property
    def std_deg(self):
        return np.rad2deg(np.sqrt(np.maximum(self.p00, 0)))
