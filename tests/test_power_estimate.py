import numpy as np
import pytest

from deskbuddy.config import load_config
from deskbuddy.planning import Plan
from deskbuddy.power_estimate import MotorAssumptions, QuasiStaticPower


def test_shoulder_pan_does_not_change_gravitational_potential():
    model = QuasiStaticPower(load_config())
    q = np.deg2rad([0, -15, 30, 15, 0])
    changed = q.copy()
    changed[0] = 0.5
    assert model.potential_j(q) == pytest.approx(model.potential_j(changed), abs=1e-6)
    assert abs(model.gravity_torque_nm(q)[0]) < 1e-5


def test_holding_power_is_nonzero_and_energy_scales_with_time():
    model = QuasiStaticPower(load_config())
    q = np.deg2rad([0, -15, 30, 15, 0])
    power, torque = model.power_w(q, np.zeros(5))
    assert np.isfinite(torque).all() and power.sum() > 0
    one = Plan(np.array([0.0, 2.0]), np.array([q, q]), model.arm.fk(q), 0.0)
    two = Plan(np.array([0.0, 4.0]), np.array([q, q]), model.arm.fk(q), 0.0)
    a, b = model.evaluate(one), model.evaluate(two)
    assert a["estimated"] and a["energy_j"] > 0
    assert b["energy_j"] == pytest.approx(2 * a["energy_j"])
    assert a["mean_power_w"] == pytest.approx(power.sum())
    assert sum(a["per_joint_energy_j"]) == pytest.approx(a["energy_j"])
    assert a["energy_wh"] == pytest.approx(a["energy_j"] / 3600)


def test_zero_mass_stationary_model_reduces_to_electronics_idle_power():
    assumptions = MotorAssumptions(voltage_v=5.4)
    model = QuasiStaticPower(load_config(), assumptions)
    model.bodies = [(0.0, com) for _, com in model.bodies]
    model.jaw_mass = 0.0
    power, _ = model.power_w(np.zeros(5), np.zeros(5))
    assert power.sum() == pytest.approx(5 * 5.4 * 0.006)
    with pytest.raises(ValueError, match="assumptions"):
        QuasiStaticPower(load_config(), MotorAssumptions(voltage_v=0))
