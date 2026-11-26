import numpy as np

from deskbuddy.state_estimation import JointKalman


def test_joint_kalman_reduces_noise_and_tracks_velocity():
    rng = np.random.default_rng(12)
    f = JointKalman([0.0])
    raw = []
    estimated = []
    for _ in range(500):
        f.predict(0.004)
        observation = rng.normal(0, np.deg2rad(0.05), 1)
        raw.append(observation[0])
        estimated.append(f.update(observation)[0])
    assert np.std(estimated[100:]) < np.std(raw[100:]) * 0.7
    initial = f.p00.copy()
    f.predict(0.2)
    assert (f.p00 > initial).all()
    ramp = JointKalman([0.0])
    for index in range(500):
        ramp.predict(0.004)
        ramp.update([np.deg2rad(6) * index * 0.004])
    assert abs(np.rad2deg(ramp.velocity[0]) - 6) < 0.1
    assert (ramp.p00 * ramp.p11 - ramp.p01**2 >= -1e-14).all()
