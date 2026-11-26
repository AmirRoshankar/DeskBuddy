import numpy as np
import pytest

from deskbuddy.feedback_loop import run_feedback
from deskbuddy.planning import Plan


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def run(
    *,
    drop_reads=False,
    drop_writes=False,
    silent_writes=False,
    persistent_reads=False,
    persistent_writes=False,
    persistent_silent=False,
    malformed_read=False,
):
    clock = Clock()
    state = {"command": np.zeros(5), "reference": np.zeros(5)}
    actual = np.zeros(5)
    goal = np.zeros(5)
    reads = 0
    writes = 0

    def read():
        nonlocal reads
        reads += 1
        if malformed_read and reads == 4:
            raise ValueError("malformed feedback packet")
        if (drop_reads and reads in (5, 6)) or (persistent_reads and reads >= 5):
            raise OSError("lost read")
        return actual.copy()

    def write(command):
        nonlocal writes, actual, goal
        writes += 1
        if (drop_writes and writes in (5, 6)) or (persistent_writes and writes >= 5):
            raise OSError("lost write")
        if (silent_writes and 5 <= writes <= 50) or persistent_silent:
            return
        goal = command.copy()
        actual = command.copy()

    plan = Plan(
        np.array([0.0, 0.5]),
        np.array([np.zeros(5), np.deg2rad([2, 0, 0, 0, 0])]),
        np.zeros(3),
        0.0,
    )
    return run_feedback(
        plan,
        np.zeros(5),
        np.full(5, 10.0),
        read,
        write,
        lambda: goal.copy(),
        lambda: False,
        state,
        lambda *args: None,
        250,
        clock.now,
        clock.sleep,
    )


def test_transient_reads_writes_and_silent_packet_loss_recover_at_250hz():
    observed, trace, maximum, stats = run(
        drop_reads=True, drop_writes=True, silent_writes=True
    )
    assert np.allclose(observed, np.deg2rad([2, 0, 0, 0, 0]), atol=np.deg2rad(0.05))
    assert stats["read_failures"] == 2 and stats["write_failures"] == 2
    assert stats["goal_mismatches"] > 0
    assert stats["requested_hz"] == 250
    commands = np.array([row["commanded_rad"] for row in trace])
    assert np.rad2deg(np.abs(np.diff(commands, axis=0))).max() <= 12 / 250 + 1e-6


@pytest.mark.parametrize(
    "kind", ["persistent_reads", "persistent_writes", "persistent_silent"]
)
def test_sustained_failure_does_not_run_blind_or_wait_forever(kind):
    with pytest.raises(
        RuntimeError, match="feedback unavailable|transmission failed|goal readback"
    ):
        run(**{kind: True})


def test_one_malformed_packet_is_retried_without_advancing_blind():
    observed, _, _, stats = run(malformed_read=True)
    assert stats["read_failures"] == 1
    assert np.allclose(observed, np.deg2rad([2, 0, 0, 0, 0]), atol=np.deg2rad(0.05))
