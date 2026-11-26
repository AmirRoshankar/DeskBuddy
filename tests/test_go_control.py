import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from deskbuddy.config import DEFAULT, load_config
from deskbuddy.go_control import launch, read_process_log
from deskbuddy.kinematics import JOINTS, Arm
from deskbuddy.planning import plan_reach


def setup():
    config = deepcopy(DEFAULT)
    config.update(calibration_verified=True, joint_mapping_verified=True)
    arm = Arm()
    seed = np.deg2rad(config["preview_start_deg"])
    plan = plan_reach(arm, seed, arm.fk(seed + [0.03, 0.02, -0.01, 0.01, 0]), config)
    return plan, config, {"torque_enabled": dict.fromkeys(JOINTS, 1)}


def test_go_launches_only_a_snapshot_of_the_selected_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, config, motor = setup()
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(poll=lambda: None)

    job = launch(plan, config, motor, popen=popen)
    snapshot = json.loads((job.directory / "selected_plan.json").read_text())
    assert snapshot["target_m"] == plan.target_m.tolist()
    assert snapshot["synthetic"] is False
    assert "--output" in calls[0]
    assert calls[0][1:3] == ["-X", "utf8"]
    assert not job.stdout.closed
    assert job.finish() is None
    assert job.stdout.closed and job.stderr.closed


def test_grasp_job_configuration_loads_in_the_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, config, motor = setup()
    request = dict(source_m=[0.2, 0.1, 0.0], max_retries=5, reference_pixel=[230, 312])
    config["grasp_request"] = request
    received = []

    def child(command, **kwargs):
        received.append(load_config(command[command.index("--config") + 1]))
        return SimpleNamespace(poll=lambda: None)

    job = launch(plan, config, motor, popen=child)
    assert received[0]["grasp_request"] == request
    job.finish()


@pytest.mark.parametrize(
    "grasp_data",
    [
        {"source_m": [0.2, 0.1]},
        {"source_m": [0.2, 0.1, float("nan")]},
        {"source_m": [0.2, 0.1, 0.0], "max_retries": 6},
        {"source_m": [0.2, 0.1, 0.0], "reference_pixel": [4]},
        {"source_m": [0.2, 0.1, 0.0], "unexpected": True},
        True,
    ],
)
def test_invalid_grasp_request_is_rejected_before_hardware(tmp_path, grasp_data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(grasp_request=grasp_data)))
    with pytest.raises(ValueError, match="grasp_request"):
        load_config(path)


def test_no_hold_or_busy_port_never_launches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, config, motor = setup()
    calls = []

    def popen(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Must not launch")

    with pytest.raises(ValueError, match="holding"):
        launch(plan, config, None, popen=popen)
    pause = tmp_path / "runs/control/pause_robot"
    pause.parent.mkdir(parents=True)
    pause.touch()
    with pytest.raises(ValueError, match="owns"):
        launch(plan, config, motor, popen=popen)
    assert not calls


def test_tracking_fault_blocks_launch_with_actionable_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, config, motor = setup()
    motor.update(
        q_model_rad=np.deg2rad([0, 10.2, 0, 0, 0]).tolist(),
        q_goal_model_rad=[0.0] * 5,
        q_reference_model_rad=[0.0] * 5,
    )
    with pytest.raises(ValueError, match="shoulder lift"):
        launch(
            plan, config, motor, popen=lambda *a, **kw: pytest.fail("Must not launch")
        )


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_current_and_legacy_degree_symbol_logs_are_readable(tmp_path, encoding):
    path = tmp_path / "stderr.log"
    message = "RuntimeError: shoulder_lift tracking error 4.04° exceeds 4.0°; stopping the reach"
    path.write_bytes(message.encode(encoding))
    assert read_process_log(path) == message


def test_redirected_worker_utf8_mode_preserves_degree_symbol(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "stderr.log"
    with path.open("wb") as stream:
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-c",
                "import sys;sys.stderr.write('tracking error 4.04\\u00b0')",
            ],
            stderr=stream,
            check=True,
        )
    assert path.read_bytes().decode("utf-8") == "tracking error 4.04°"
