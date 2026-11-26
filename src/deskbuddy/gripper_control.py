"""Shared bounded gripper ramp for explicit grasp and Home commands."""

import time

import numpy as np


def ramp_gripper(bus, percent, stopped, sleep=time.sleep):
    if stopped():
        raise RuntimeError("Grab stopped")
    mode = bus.sync_read("Operating_Mode", ["gripper"], normalize=False, num_retry=2)[
        "gripper"
    ]
    if mode != 0:
        raise ValueError("Gripper must be in position mode")
    current = float(
        bus.sync_read("Present_Position", ["gripper"], num_retry=2)["gripper"]
    )
    enabled = bus.sync_read("Torque_Enable", ["gripper"], normalize=False, num_retry=2)[
        "gripper"
    ]
    if not enabled:
        bus.sync_write("Goal_Position", {"gripper": current}, num_retry=2)
        goal = bus.sync_read("Goal_Position", ["gripper"], num_retry=2)["gripper"]
        if abs(goal - current) > 2:
            raise ValueError("Gripper hold target did not verify; torque remains off")
        if stopped():
            raise RuntimeError("Grab stopped")
        bus.enable_torque(["gripper"], num_retry=2)
    blocked = 0
    for goal in np.linspace(
        current, percent, max(2, int(abs(percent - current) / 2) + 1)
    ):
        if stopped():
            raise RuntimeError("Grab stopped; gripper hold retained")
        temperature = bus.sync_read(
            "Present_Temperature", ["gripper"], normalize=False, num_retry=2
        )["gripper"]
        if temperature >= 50:
            raise RuntimeError("Gripper temperature limit; hold retained")
        position = bus.sync_read("Present_Position", ["gripper"], num_retry=2)[
            "gripper"
        ]
        blocked = blocked + 1 if percent < current and position - goal > 8 else 0
        if blocked >= 3:
            break
        bus.sync_write("Goal_Position", {"gripper": float(goal)}, num_retry=2)
        sleep(0.05)
    position = float(
        bus.sync_read("Present_Position", ["gripper"], num_retry=2)["gripper"]
    )
    return dict(gripper_percent=position, gripper_closed=position <= 5)
