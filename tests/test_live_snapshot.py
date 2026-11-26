import json

import numpy as np

from deskbuddy import live_snapshot


def test_snapshot_write_and_locked_destination(tmp_path, monkeypatch):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4))
    assert live_snapshot.publish(tmp_path, rgb, depth, {"tick": 1}) == {}
    assert json.loads((tmp_path / "state.json").read_text()) == {"tick": 1}
    replace = live_snapshot.os.replace

    def locked(source, destination):
        if destination.name == "state.json":
            raise PermissionError("simulated Windows sharing lock")
        return replace(source, destination)

    monkeypatch.setattr(live_snapshot.os, "replace", locked)
    errors = live_snapshot.publish(tmp_path, rgb, depth, {"tick": 2})
    assert list(errors) == ["state.json"]
    assert json.loads((tmp_path / "state.json").read_text()) == {"tick": 1}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))
