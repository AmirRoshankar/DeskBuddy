import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .geometry import fit_rigid
from .kinematics import Arm
from .planning import plan_reach
from .storage import save_plan


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def doctor(require_cuda=False):
    from importlib.metadata import version

    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {},
    }
    ok = True
    for package in (
        "numpy",
        "scipy",
        "Pillow",
        "pyrealsense2",
        "lerobot",
        "torch",
        "torchvision",
    ):
        try:
            report["packages"][package] = version(package)
        except Exception:
            report["packages"][package] = "not installed"
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_runtime"] = torch.version.cuda
        if report["cuda_available"]:
            device = torch.device("cuda")
            report["gpu"] = torch.cuda.get_device_name()
            report["vram_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 2
            )
            report["compute_capability"] = list(torch.cuda.get_device_capability())
            # Verify kernels, gradients and optimizer updates, not just device discovery.
            torch.manual_seed(7)
            model = torch.nn.Sequential(
                torch.nn.Linear(3, 64), torch.nn.SiLU(), torch.nn.Linear(64, 3)
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            x = torch.randn(128, 3, device=device)
            y = 0.5 * x
            initial = float(torch.nn.functional.mse_loss(model(x), y).detach().cpu())
            for _ in range(40):
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            final = float(torch.nn.functional.mse_loss(model(x), y).detach().cpu())
            if not np.isfinite(final) or final >= initial:
                raise RuntimeError("GPU optimization smoke test did not converge")
            report["gpu_training_smoke"] = {
                "initial_mse": initial,
                "final_mse": final,
                "passed": True,
            }
        elif require_cuda:
            ok = False
    except Exception as exc:
        report["gpu_error"] = str(exc)
        ok = not require_cuda
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        report["lerobot_import"] = bool(SO101Follower and SO101FollowerConfig)
    except Exception as exc:
        report["lerobot_import_error"] = str(exc)
        ok = False
    print(json.dumps(report, indent=2))
    write_json("runs/environment.json", report)
    return 0 if ok else 1


def devices():
    """Enumerate devices only; do not open serial buses or configure motors."""
    from serial.tools import list_ports

    report = {
        "serial_ports": [
            {"port": p.device, "description": p.description}
            for p in list_ports.comports()
        ]
    }
    try:
        import pyrealsense2 as rs

        report["cameras"] = [
            {
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
            }
            for d in rs.context().query_devices()
        ]
    except Exception as exc:
        report["camera_error"] = str(exc)
    print(json.dumps(report, indent=2))


def headless_demo(config, output):
    from PIL import Image, ImageDraw

    from .demo import arm_image, synthetic_frame, synthetic_point

    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    target = synthetic_point(385, 195) + [0, 0, config["hover_m"]]
    start = np.deg2rad(config["preview_start_deg"])
    began = time.perf_counter()
    plan = plan_reach(arm, start, target, config)
    elapsed = (time.perf_counter() - began) * 1000
    output = Path(output)
    save_plan(output / "demo_plan.json", plan, config, synthetic=True)
    image = Image.new("RGB", (1140, 540), "#0e1820")
    image.paste(synthetic_frame()[0], (0, 0))
    ImageDraw.Draw(image).ellipse((378, 188, 392, 202), outline="white", width=2)
    image.paste(arm_image(arm, plan.joints_rad[-1], target, plan), (640, 0))
    ImageDraw.Draw(image).text(
        (20, 498),
        f"OFFLINE FIXTURE | Endpoint model error: {plan.endpoint_error_m * 1000:.2f} mm | "
        f"Duration: {plan.time_s[-1]:.1f} s | Hardware measurements pending",
        fill="white",
    )
    image.save(output / "demo.png")
    report = {
        "synthetic": True,
        "target_m": target.tolist(),
        "planning_ms": elapsed,
        "endpoint_model_error_mm": plan.endpoint_error_m * 1000,
        "duration_s": float(plan.time_s[-1]),
        "samples": len(plan.time_s),
    }
    write_json(output / "demo_metrics.json", report)
    print(json.dumps(report, indent=2))


def benchmark(config, output):
    arm = Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m"))
    rng = np.random.default_rng(42)
    start = np.deg2rad(config["preview_start_deg"])
    rows = []
    # Targets come from known-valid model poses; this does not measure real accuracy.
    for _ in range(30):
        goal_pose = start + np.deg2rad(
            rng.uniform([-15, -8, -10, -8, -10], [15, 8, 10, 8, 10])
        )
        target = arm.fk(goal_pose)
        begin = time.perf_counter()
        try:
            p = plan_reach(arm, start, target, config)
            rows.append(
                {
                    "valid": True,
                    "error_mm": p.endpoint_error_m * 1000,
                    "planning_ms": (time.perf_counter() - begin) * 1000,
                }
            )
        except ValueError as exc:
            rows.append({"valid": False, "reason": str(exc)})
    passed = [row for row in rows if row["valid"]]
    report = {
        "synthetic": True,
        "seed": 42,
        "trials": len(rows),
        "valid": len(passed),
        "mean_endpoint_model_error_mm": float(np.mean([r["error_mm"] for r in passed]))
        if passed
        else None,
        "median_planning_ms": float(np.median([r["planning_ms"] for r in passed]))
        if passed
        else None,
        "trials_detail": rows,
    }
    write_json(output, report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "trials_detail"}, indent=2)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DeskBuddy robotics workbench. Preview commands never move hardware."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser(
        "doctor", help="Check imports and run a tiny GPU training smoke test"
    )
    d.add_argument("--require-cuda", action="store_true")
    sub.add_parser(
        "devices",
        help="Enumerate cameras and serial ports without connecting to motors",
    )
    for name in ("demo", "camera", "benchmark", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--config")
        if name == "demo":
            p.add_argument("--headless", action="store_true")
            p.add_argument("--output", default="runs/offline")
        if name == "benchmark":
            p.add_argument("--output", default="runs/benchmark.json")
        if name == "camera":
            p.add_argument(
                "--robot-state",
                action="store_true",
                help="Read calibrated motor state without torque or motion writes",
            )
        if name == "plan":
            p.add_argument(
                "--target", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
            )
            p.add_argument("--output", default="runs/plan.json")
    p = sub.add_parser(
        "calibrate",
        help="Fit camera-to-base transform from measured 3D correspondences",
    )
    p.add_argument("points", help="JSON with camera_points_m and base_points_m")
    p.add_argument("--config", help="Existing settings to retain")
    p.add_argument("--output", default="configs/local.json")
    p.add_argument("--max-error-mm", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.require_cuda)
        if args.command == "devices":
            devices()
            return 0
        config = load_config(args.config)
        if args.command == "demo" and args.headless:
            headless_demo(config, args.output)
        elif args.command in ("demo", "camera"):
            from .gui import run

            run(
                config,
                live=args.command == "camera",
                robot_state=getattr(args, "robot_state", False),
                config_path=args.config,
            )
        elif args.command == "benchmark":
            benchmark(config, args.output)
        elif args.command == "plan":
            plan = plan_reach(
                Arm(config.get("model_joint_limits_deg"), config.get("tool_offset_m")),
                np.deg2rad(config["preview_start_deg"]),
                args.target,
                config,
            )
            save_plan(args.output, plan, config, synthetic=True)
            print(
                f"Saved preview plan to {args.output}; model error {plan.endpoint_error_m * 1000:.2f} mm"
            )
        elif args.command == "calibrate":
            if not np.isfinite(args.max_error_mm) or args.max_error_mm <= 0:
                raise ValueError("Calibration error threshold must be positive")
            points = json.loads(Path(args.points).read_text(encoding="utf-8"))
            t, errors = fit_rigid(points["camera_points_m"], points["base_points_m"])
            if errors.max() * 1000 > args.max_error_mm:
                raise ValueError(
                    f"Calibration rejected: maximum fit error {errors.max() * 1000:.1f} mm"
                )
            config["T_base_camera"] = t.tolist()
            config["calibration_verified"] = False
            write_json(args.output, config)
            print(
                f"Saved transform. RMS fit error {np.sqrt(np.mean(errors**2)) * 1000:.2f} mm. "
                "Verify against independent measured points before setting calibration_verified=true."
            )
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
