import os
import sys

from deskbuddy import desktop_launcher


def test_duplicate_launcher_does_not_open_camera_or_robot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "runs/live/process_id.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("1234")
    monkeypatch.setattr(desktop_launcher, "process_running", lambda pid: True)
    monkeypatch.setattr(
        desktop_launcher,
        "main",
        lambda: (_ for _ in ()).throw(AssertionError("No second UI")),
    )
    desktop_launcher.launch()
    assert marker.read_text() == "1234"


def test_launcher_preserves_disable_and_only_selects_read_only_startup(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/local.json").write_text("{}")
    disabled = tmp_path / "runs/control/hardware_disabled"
    disabled.parent.mkdir(parents=True)
    disabled.write_text("operator disabled")
    calls = []
    monkeypatch.setattr(sys, "argv", [])
    monkeypatch.setattr(desktop_launcher, "main", lambda: calls.append(sys.argv.copy()))
    desktop_launcher.launch()
    assert calls == [
        ["deskbuddy", "camera", "--config", "configs/local.json", "--robot-state"]
    ]
    assert disabled.read_text() == "operator disabled"
    assert not (tmp_path / "runs/live/process_id.txt").exists()
    assert desktop_launcher.process_running(os.getpid())
