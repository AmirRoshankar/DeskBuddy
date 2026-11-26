"""Shared tracking diagnostics for UI gating and readable failure messages."""

import numpy as np

from .kinematics import JOINTS

TRACKING_LIMIT_DEG = 10.0


def running_tracking_limits(initial_error_deg):
    """Unexpected error against the physical reference, after fixed load correction."""
    initial = np.asarray(initial_error_deg, float)
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError("Expected five finite initial tracking errors")
    return np.full(5, TRACKING_LIMIT_DEG)


def tracking_fault(motor_state):
    if not motor_state or "q_goal_model_rad" not in motor_state:
        return None
    actual = np.asarray(motor_state.get("q_model_rad"), float)
    goal = np.asarray(motor_state["q_goal_model_rad"], float)
    if (
        actual.shape != (5,)
        or goal.shape != (5,)
        or not np.isfinite(actual).all()
        or not np.isfinite(goal).all()
    ):
        return "Joint feedback is invalid"
    reference = motor_state.get("q_reference_model_rad")
    if reference is None:
        # Static holding deflection is measured and compensated on Go. It is
        # not evidence of a failed new movement and must not latch out every Go.
        return None
    reference = np.asarray(reference, float)
    if reference.shape != (5,) or not np.isfinite(reference).all():
        return "Physical reference feedback is invalid"
    error = np.rad2deg(np.abs(actual - reference))
    limits = np.asarray(
        motor_state.get("tracking_limits_deg", [TRACKING_LIMIT_DEG] * 5), float
    )
    if limits.shape != (5,) or not np.isfinite(limits).all():
        return "Tracking-limit feedback is invalid"
    index = int(np.argmax(error - limits))
    if error[index] > limits[index]:
        return f"{JOINTS[index].replace('_', ' ')} new motion error {error[index]:.1f}° exceeds {limits[index]:.1f}°; motion stopped"
    return None
