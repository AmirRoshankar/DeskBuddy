import pytest

from deskbuddy.timing_benchmark import benchmark


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_read_only_benchmark_distinguishes_headroom_and_deadline_misses():
    clock = Clock()

    class Bus:
        def sync_read(self, register, names, num_retry=0):
            assert register in ("Present_Position", "Goal_Position")
            clock.sleep(0.006)
            return dict.fromkeys(names, 0.0)

    report = benchmark(
        Bus(), rates=(20, 100), seconds=0.2, clock=clock.now, sleep=clock.sleep
    )
    assert report["candidate_hz"] == 20
    assert not report["position_commands_sent"] and not report["torque_commands_sent"]
    pair100 = report["sweeps"][-1]
    assert pair100["missed_deadlines"] > 0
    assert pair100["latency"]["median_ms"] == pytest.approx(12.0)
