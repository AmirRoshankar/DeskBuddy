from copy import deepcopy

import numpy as np
import pytest

from deskbuddy.config import DEFAULT
from deskbuddy.hardware_interlock import (
    disable_motion,
    enable_motion,
    motion_disabled,
    require_hardware_enabled,
)
from deskbuddy.planning import check_workspace
from deskbuddy.ui.workspace import edited_workspace


def test_disable_is_persistent_blocks_launch_and_enable_does_not_clear_stop(
    tmp_path, monkeypatch
):
    from deskbuddy.go_control import launch

    monkeypatch.chdir(tmp_path)
    disable_motion()
    assert motion_disabled()
    assert (tmp_path / "runs/control/stop").exists()
    with pytest.raises(RuntimeError, match="Hardware disabled"):
        launch(None, None, None, popen=lambda *a, **k: pytest.fail("No worker allowed"))
    enable_motion()
    require_hardware_enabled()
    assert not motion_disabled()
    assert (tmp_path / "runs/control/stop").exists()


def test_workspace_editor_converts_cm_without_mutating_current_config():
    before = deepcopy(DEFAULT)
    result = edited_workspace(before, [10, -20, 3], [40, 20, 30], False, True)
    assert result["workspace_min_m"] == [0.1, -0.2, 0.03]
    assert result["workspace_max_m"] == [0.4, 0.2, 0.3]
    assert before == DEFAULT
    assert not result["restrict_to_test_bed"]
    for low, high in (([0, 0, 0], [1, 0, 1]), ([0, 0, 0], [1, np.nan, 1])):
        with pytest.raises(ValueError):
            edited_workspace(before, low, high, False, True)


def test_box_mode_does_not_reject_valid_targets_outside_the_paper():
    config = deepcopy(DEFAULT)
    config["test_bed"] = {
        "corners_base_m": [
            [0.1, -0.1, 0],
            [0.3, -0.1, 0],
            [0.3, 0.1, 0],
            [0.1, 0.1, 0],
        ],
        "inset_m": 0.02,
    }
    point = [0.35, 0, 0.1]
    with pytest.raises(ValueError, match="A4"):
        check_workspace(point, config)
    config["restrict_to_test_bed"] = False
    check_workspace(point, config)
    with pytest.raises(ValueError, match="outside workspace"):
        check_workspace([0.6, 0, 0.1], config)


def test_support_clearance_pending_blocks_any_new_reach(tmp_path, monkeypatch):
    from deskbuddy.go_control import launch

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "runs/control/support_clearance_required"
    marker.parent.mkdir(parents=True)
    marker.touch()
    with pytest.raises(ValueError, match="hands/support clear"):
        launch(
            None,
            None,
            None,
            popen=lambda *a, **k: pytest.fail("No reach before support clearance"),
        )
