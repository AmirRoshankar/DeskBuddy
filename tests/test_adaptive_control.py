import numpy as np

from deskbuddy.adaptive_control import AdaptiveCorrection
from deskbuddy.feedback_loop import run_feedback
from deskbuddy.planning import Plan


def test_varying_load_bias_is_corrected_instead_of_frozen_for_entire_reach():
    now = [0.0]
    goal = np.deg2rad([-3.0, 0, 0, 0, 0])

    def clock():
        return now[0]

    def sleep(dt):
        now[0] += dt

    def read():
        load = 3.0 - 2.8 * min(now[0] / 4.0, 1.0)
        return goal + np.deg2rad([load, 0, 0, 0, 0])

    def write(command):
        nonlocal goal
        goal = command.copy()

    plan = Plan(
        np.array([0.0, 4.0]),
        np.deg2rad([[0, 0, 0, 0, 0], [8, 0, 0, 0, 0]]),
        np.zeros(3),
        0.0,
    )
    controller = {"command": goal.copy(), "reference": np.zeros(5)}
    observed, trace, _, stats = run_feedback(
        plan,
        np.deg2rad([3, 0, 0, 0, 0]),
        np.full(5, 10.0),
        read,
        write,
        lambda: goal.copy(),
        lambda: False,
        controller,
        lambda *args: None,
        250,
        clock,
        sleep,
    )
    assert abs(np.rad2deg(observed[0]) - 8) < 0.25
    assert abs(np.rad2deg(trace[-1]["correction_rad"][0])) < 0.5
    assert stats["adaptive_correction"] and stats["settling_converged"]
    assert stats["max_sent_speed_deg_s"] <= 12 + 1e-8


def test_rate_and_joint_limits_prevent_windup_and_keep_locked_axis_unchanged():
    c = AdaptiveCorrection(
        np.zeros(5),
        np.zeros(5),
        np.full(5, -0.2),
        np.full(5, 0.2),
        locked=[False] * 4 + [True],
    )
    previous = np.zeros(5)
    for _ in range(1000):
        command, integral, _ = c.propose(np.ones(5), np.zeros(5), previous, 0.004)
        assert np.abs(command - previous).max() <= np.deg2rad(12) * 0.004 + 1e-12
        assert (command <= 0.2).all() and command[4] == 0
        c.commit(integral)
        previous = command
    assert np.abs(c.integral).max() < 0.01


def test_feedback_remains_stable_with_servo_lag_noisy_encoders_and_changing_load():
    rng = np.random.default_rng(23)
    now = [0.0]
    last = [0.0]
    actual = np.zeros(5)
    goal = np.deg2rad([-3.0, 0, 0, 0, 0])

    def clock():
        return now[0]

    def sleep(dt):
        now[0] += dt

    def read():
        nonlocal actual
        dt = now[0] - last[0]
        last[0] = now[0]
        load = np.deg2rad([3.0 - 2.8 * min(now[0] / 4.0, 1.0), 0, 0, 0, 0])
        actual += (goal + load - actual) * (1 - np.exp(-dt / 0.06))
        return actual + rng.normal(0, np.deg2rad(0.025), 5)

    def write(command):
        nonlocal goal
        goal = command.copy()

    plan = Plan(
        np.array([0.0, 4.0]),
        np.deg2rad([[0, 0, 0, 0, 0], [8, 0, 0, 0, 0]]),
        np.zeros(3),
        0.0,
    )
    _, _, maximum, stats = run_feedback(
        plan,
        np.deg2rad([3, 0, 0, 0, 0]),
        np.full(5, 10.0),
        read,
        write,
        lambda: goal.copy(),
        lambda: False,
        {"command": goal.copy(), "reference": np.zeros(5)},
        lambda *args: None,
        250,
        clock,
        sleep,
    )
    assert abs(np.rad2deg(actual[0]) - 8) < 0.25
    assert np.rad2deg(maximum).max() < 2
    assert stats["max_sent_speed_deg_s"] <= 12 + 1e-8
