"""Windows Tk integration test: clicks preview; only Go may launch motion."""

import json
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows Tk integration test"
)


def test_click_go_gating_and_invalid_selection(tmp_path, monkeypatch):
    import tkinter as tk

    from deskbuddy import camera, go_control, grasp_keys, gui, passive_arm
    from deskbuddy.config import load_config
    from deskbuddy.demo import T_BASE_CAMERA
    from deskbuddy.planning import plan_reach
    from deskbuddy.trajectory import execution, yolo_perception

    monkeypatch.setattr(
        "deskbuddy.trajectory.path_overlays.OverlayPlanner.compare",
        lambda *args, **kwargs: [],
    )

    class Detector:
        def update(self, *args):
            return [], dict(
                scene_revision=7, change_pending=False, detection_unix_s=time.time()
            )

        def close(self):
            pass

    def fake_route(
        self, arm, start, target, config, boxes, selected, candidates, revision
    ):
        plan = plan_reach(arm, start, target, config)
        plan.planner_info = [
            dict(
                trajectory_method=selected,
                candidates=candidates,
                scene_revision=revision,
                obstacles_m=[],
                fallback=False,
                allow_object_collisions=config.get("allow_object_collisions", False),
            )
        ]
        return plan

    monkeypatch.setattr(yolo_perception, "YoloService", Detector)
    monkeypatch.setattr(execution.RoutePlanner, "plan", fake_route)

    monkeypatch.setattr(
        grasp_keys,
        "detect_key_target",
        lambda *a: (np.array([0.25, 0, 0]), dict(pixel_roi=[380, 190, 390, 200])),
    )

    monkeypatch.chdir(tmp_path)
    config = load_config()
    config.update(
        calibration_verified=True,
        joint_mapping_verified=True,
        T_base_camera=T_BASE_CAMERA.tolist(),
        trajectory_method="direct",
    )
    q = np.deg2rad(config["preview_start_deg"]).tolist()
    q_goal = np.array(q)
    q_goal[2] -= 0.03
    calls = []
    errors = []

    class Camera:
        def __init__(self, serial):
            self.frames = 0

        def read(self):
            self.frames += 1
            if 3 <= self.frames <= 5:
                raise RuntimeError("temporary camera dropout")
            return (
                np.zeros((480, 640, 3), np.uint8),
                np.full((480, 640), 0.65 if self.frames == 1 else 0.0),
                SimpleNamespace(fx=600, fy=600, ppx=320, ppy=240),
            )

        def point(self, depth, intrinsics, u, v):
            raise AssertionError(
                "Preview and Go must use the scene estimate, not require fresh raw depth"
            )

        def close(self):
            pass

    class Passive:
        def __init__(self, config):
            pass

        def read(self):
            return {
                "q_model_rad": q,
                "q_goal_model_rad": q_goal.tolist(),
                "gripper_percent": 0,
                "torque_enabled": {
                    name: 1
                    for name in (
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    )
                },
                "voltage_v": {"elbow_flex": 5.4},
                "temperature_c": {"elbow_flex": 30},
                "telemetry_unix_s": time.time(),
            }

        def close(self):
            pass

        def resume(self):
            pass

    def fake_popen(command, **kwargs):
        from pathlib import Path

        calls.append(command)
        folder = Path(command[command.index("--output") + 1])
        (folder / "result.json").write_text(
            json.dumps({"completed": True, "settled_model_endpoint_error_mm": 1.5})
        )
        return SimpleNamespace(poll=lambda: 0)

    def all_widgets(widget):
        result = [widget]
        for child in widget.winfo_children():
            result.extend(all_widgets(child))
        return result

    original_tk = tk.Tk
    drawn = []
    from deskbuddy.ui import scene_inspector

    original_panel = scene_inspector.render_scene
    scene_frames = []

    def measured_panel(snapshot, *args, **kwargs):
        drawn.append(np.asarray(snapshot["q"]).copy())
        scene_frames.append((snapshot.get("plan"), list(snapshot.get("paths", []))))
        return original_panel(snapshot, *args, **kwargs)

    monkeypatch.setattr(scene_inspector, "render_scene", measured_panel)

    def hidden_root():
        root = original_tk()
        root.withdraw()
        root.report_callback_exception = lambda *args: (
            errors.append(args),
            traceback.print_exception(*args),
        )

        def button():
            return next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Button) and w.cget("text") == "Go"
            )

        def click(u, v):
            from deskbuddy.kinematics import Arm
            from deskbuddy.ui.scene_inspector import OrbitView, scene_projection

            scene = next(w for w in all_widgets(root) if w.winfo_name() == "scene_view")
            size = (max(160, scene.winfo_width()), max(120, scene.winfo_height()))
            center, scale = scene_projection(dict(arm=Arm(), q=q, config=config), size)
            target = [0.25, 0, config["hover_m"]] if u else [2, 2, config["hover_m"]]
            x, y = OrbitView().project([target], center, scale, size)[0]
            for binding in ("<ButtonPress-1>", "<ButtonRelease-1>"):
                name = scene.bind(binding).split("[", 1)[1].split()[0]
                values = ["??"] * 19
                values[0] = values[1] = "1"
                values[5] = "0"
                values[8] = str(int(round(x)))
                values[9] = str(int(round(y)))
                root.tk.call(name, *values)

        def select_and_go():
            assert not calls
            assert button().cget("state") == "disabled"
            assert button().cget("foreground") == "white"
            assert button().cget("disabledforeground") == "white"
            assert "bold" in button().cget("font")
            click(385, 195)
            root.after(250, check_preview)

        preview_deadline = time.monotonic() + 5

        def check_preview():
            if (
                not (tmp_path / "runs/last_plan.json").exists()
                and time.monotonic() < preview_deadline
            ):
                root.after(50, check_preview)
                return
            assert not calls, "Selecting a point must not execute motion"
            assert np.allclose(drawn[-1], q), (
                "Selecting a target must not animate the measured arm"
            )
            snapshot = json.loads((tmp_path / "runs/last_plan.json").read_text())
            assert np.allclose(snapshot["joints_rad"][0], q), (
                "The physical reference starts at the measured pose"
            )
            assert button().cget("state") == "normal"
            scene_button = next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Button) and w.cget("text") == "Pop out 3D"
            )
            scene_button.invoke()
            assert any(
                isinstance(w, tk.Button) and w.cget("text") == "Home"
                for w in all_widgets(root)
            )
            assert any(
                isinstance(w, tk.Toplevel) and w.title().startswith("3D scene")
                for w in all_widgets(root)
            )
            override = next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Checkbutton)
                and w.cget("text").startswith("Allow predicted")
            )
            override.invoke()
            assert config["allow_object_collisions"]
            from deskbuddy.ui.hover import HoverControl

            menu = next(w for w in all_widgets(root) if isinstance(w, tk.OptionMenu))
            menu["menu"].invoke(5)
            radio = next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Radiobutton) and w.cget("text") == "4"
            )
            radio.invoke()
            hover = next(w for w in all_widgets(root) if isinstance(w, HoverControl))
            before = config["hover_m"]
            hover.adjust(120)
            assert config["hover_m"] == pytest.approx(before + 0.002)
            assert button().cget("state") == "disabled"
            assert not calls, "Wheel adjustment must not start motion"

            def go_after_preview():
                assert button().cget("state") == "normal"
                updated = json.loads((tmp_path / "runs/last_plan.json").read_text())
                assert updated["target_m"][2] == pytest.approx(config["hover_m"])
                assert updated["planner_info"][0]["trajectory_method"] == "imf"
                assert updated["planner_info"][0]["candidates"] == 4
                assert updated["planner_info"][0]["allow_object_collisions"]
                button().invoke()
                assert len(calls) == 1, [
                    w.get("1.0", "end")
                    for w in all_widgets(root)
                    if isinstance(w, tk.Text)
                ]
                root.after(250, after_motion)

            root.after(350, go_after_preview)

        def invalid_selection():
            assert button().cget("state") == "disabled"
            click(0, 0)
            assert button().cget("state") == "disabled"
            button().invoke()
            assert len(calls) == 1, "Invalid selection must not reuse the old target"
            next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Button) and w.cget("text") == "Grab keys"
            ).invoke()
            next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Button) and w.cget("text") == "Stop"
            ).invoke()

            def verify_cancelled():
                assert len(calls) == 1, (
                    "Stop during automatic-grab planning must prevent a later launch"
                )

            root.after(250, verify_cancelled)
            root.after(400, recalibrate)
            root.after(700, root.destroy)

        def after_motion():
            if button().cget("state") != "disabled":
                root.after(75, after_motion)
                return
            assert scene_frames[-1] == (None, []), (
                "Completed motion must clear all rendered paths"
            )
            invalid_selection()

        def recalibrate():
            reset = next(
                w
                for w in all_widgets(root)
                if isinstance(w, tk.Button) and w.cget("text") == "Recalibrate"
            )
            reset.invoke()
            assert (
                not config["calibration_verified"] and config["T_base_camera"] is None
            )
            assert button().cget("state") == "disabled"
            saved = json.loads((tmp_path / "local.json").read_text())
            assert not saved["calibration_verified"]
            assert len(calls) == 1, "Recalibration must never launch robot motion"

        root.after(850, select_and_go)
        root.after(12000, root.destroy)
        return root

    monkeypatch.setattr(camera, "RealSense", Camera)
    monkeypatch.setattr(passive_arm, "PassiveArm", Passive)
    monkeypatch.setattr(go_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tk, "Tk", hidden_root)
    gui.run(config, live=True, robot_state=True, config_path=tmp_path / "local.json")
    assert not errors, errors
    assert len(calls) == 1
    assert all(np.allclose(pose, q) for pose in drawn), (
        "Live arm drawing must remain tied to encoders"
    )
