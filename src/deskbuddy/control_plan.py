"""Physical joint path and initial bias for a continuous, adaptive command start."""

import numpy as np

from .kinematics import JOINTS
from .planning import halve_steps, plan_reach, validate_plan


def arm_holding(motor_state):
    return bool(motor_state) and all(
        motor_state.get("torque_enabled", {}).get(j) == 1 for j in JOINTS
    )


def preview_from_feedback(arm, measured, motor_state, target, config):
    if not arm_holding(motor_state):
        # Goal registers can reset to zero after power loss. They are not a
        # hold target while torque is off; only measured joints define preview.
        return plan_reach(arm, measured, target, config)
    held = np.asarray(motor_state["q_goal_model_rad"], float)
    return compensated_plan(arm, measured, held, target, config)[0]


def compensated_plan(arm, measured, held_command, target, config):
    measured = np.asarray(measured, float)
    held_command = np.asarray(held_command, float)
    if (
        measured.shape != (5,)
        or held_command.shape != (5,)
        or not np.isfinite(measured).all()
        or not np.isfinite(held_command).all()
    ):
        raise ValueError("Need five finite measured and commanded joint angles")
    bias = measured - held_command
    planning_config = dict(config, control_hz=min(20.0, config["control_hz"]))
    plan = validate_plan(
        arm, halve_steps(plan_reach(arm, measured, target, planning_config)), config
    )
    commands = plan.joints_rad - bias
    if (held_command < arm.lower).any() or (held_command > arm.upper).any():
        raise ValueError(arm.limit_error(held_command, "Existing hold target"))
    if not np.allclose(commands[0], held_command, atol=1e-10):
        raise ValueError("Initial command would not match the existing hold")
    return plan, commands, bias
