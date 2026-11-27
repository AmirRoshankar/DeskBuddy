from deskbuddy.gripper_control import ramp_gripper


class Bus:
    """Gripper that stops at 30% (an object between the jaws)."""

    def __init__(self):
        self.position, self.goals = 80.0, []

    def sync_read(self, name, motors, normalize=True, num_retry=0):
        value = dict(
            Operating_Mode=0,
            Present_Position=self.position,
            Torque_Enable=1,
            Present_Temperature=30,
            Goal_Position=self.goals[-1] if self.goals else self.position,
        )[name]
        return {"gripper": value}

    def sync_write(self, name, values, num_retry=0):
        goal = values["gripper"]
        self.goals.append(goal)
        self.position = max(30.0, goal)

    def enable_torque(self, motors, num_retry=0):
        pass


def test_close_continues_to_the_requested_goal_after_contact():
    bus = Bus()
    result = ramp_gripper(bus, 2.0, lambda: False, sleep=lambda s: None)
    assert result["gripper_contact"]
    assert bus.goals[-1] == 2.0
    assert result["gripper_percent"] == 30.0
