"""The optional motor diagnostic must never invoke a configuration/torque write."""

import pytest

pytest.importorskip("lerobot")
from deskbuddy import inspect_bus


@pytest.mark.parametrize("read_fails", [False, True])
def test_probe_only_reads_and_closes_without_torque_change(monkeypatch, read_fails):
    calls = []

    class ReadOnlyBus:
        def __init__(self, port, motors):
            assert port == "COM_TEST"
            assert [m.id for m in motors.values()] == list(range(1, 7))

        def connect(self, *, handshake):
            calls.append(("connect", handshake))

        def ping(self, motor_id, *, num_retry):
            calls.append(("read_model", motor_id))
            if read_fails:
                raise OSError("Disconnected during read")
            return 777

        def disconnect(self, *, disable_torque):
            calls.append(("disconnect", disable_torque))

    monkeypatch.setattr(inspect_bus, "SerialMotorsBus", ReadOnlyBus)
    if read_fails:
        with pytest.raises(OSError):
            inspect_bus.inspect("COM_TEST")
    else:
        report = inspect_bus.inspect("COM_TEST")
        assert report["all_expected_ids_respond"]
        assert len([call for call in calls if call[0] == "read_model"]) == 12
    assert calls[0] == ("connect", False)
    assert calls[-1] == ("disconnect", False)
    assert all(call[0] in ("connect", "read_model", "disconnect") for call in calls)
