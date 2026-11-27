"""One operator-authorized, small comparison trial through the normal motion worker."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .calibration_session import write_json
from .config import load_config
from .control_plan import compensated_plan
from .go_control import launch
from .hardware_interlock import require_hardware_enabled
from .kinematics import Arm
from .learned.inference import get_proposal
from .storage import save_plan


def fresh_state():
    value = json.loads(Path("runs/live/state.json").read_text(encoding="utf-8"))
    if (
        time.time() - value["unix_time_s"] > 1.5
        or value.get("robot_io_paused")
        or value.get("camera_stale")
        or value.get("motor_error")
        or value.get("motor_state") is None
    ):
        raise ValueError("Need fresh, unpaused camera and motor state")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--model", choices=("imf", "mlp"), required=True)
    parser.add_argument(
        "--target-m", required=True, help="Comma-separated XYZ in metres"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute this one checked trial; requires operator authorization",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(
        planner_mode="flow_assisted",
        proposal_method=args.model,
        flow_checkpoint=f"runs/deskbuddy/{args.model}.pt",
        max_joint_speed_deg_s=min(12.0, config["max_joint_speed_deg_s"]),
    )
    target = np.fromstring(args.target_m, sep=",")
    if target.shape != (3,) or not np.isfinite(target).all():
        parser.error("Target must have three finite coordinates")
    require_hardware_enabled()
    # Load before reading the trial start so PyTorch startup does not stale it.
    proposal = get_proposal(config["flow_checkpoint"], config, method=args.model)
    live = fresh_state()
    motor = live["motor_state"]
    measured = np.array(motor["q_model_rad"])
    held = np.array(motor["q_goal_model_rad"])
    arm = Arm(config["model_joint_limits_deg"], config["tool_offset_m"])
    if np.linalg.norm(target - arm.fk(measured)) > 0.03:
        raise ValueError("Comparison trial exceeds 3 cm from measured start")
    # Warm the proposal's normal execution path, then independently plan/check.
    proposal.predict(measured, target)
    began = time.perf_counter()
    plan, _, _ = compensated_plan(arm, measured, held, target, config)
    planning_ms = (time.perf_counter() - began) * 1000
    if np.rad2deg(np.abs(plan.joints_rad - measured)).max() > 8:
        raise ValueError("Comparison trial exceeds 8 degrees from measured start")
    if not plan.planner_info or any(
        p.get("used") != "flow_assisted" or p.get("proposal_method") != args.model
        for p in plan.planner_info
    ):
        raise ValueError(
            "Learned proposal fell back to numerical IK; choose another comparison target"
        )
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    save_plan(folder / "preview.json", plan, config, synthetic=False)
    shutil.copyfile("runs/live/rgb.png", folder / "before.png")
    manifest = dict(
        model=args.model,
        target_m=target.tolist(),
        start_model_rad=measured.tolist(),
        planning_ms=planning_ms,
        planner_info=plan.planner_info,
        preview_duration_s=float(plan.time_s[-1]),
        executed=False,
        independent_visual_accuracy_measured=False,
    )
    write_json(folder / "trial.json", manifest)
    print(json.dumps(manifest), flush=True)
    if not args.execute:
        return
    fresh_state()  # Final read-only check before asking the standard worker to act.
    job = launch(plan, config, motor)
    manifest.update(
        executed=True, job_directory=str(job.directory), worker_pid=job.process.pid
    )
    write_json(folder / "trial.json", manifest)
    try:
        code = job.process.wait()
    except KeyboardInterrupt:
        Path("runs/control/stop").touch()
        job.process.wait(timeout=10)
        job.finish()
        raise
    result = job.finish()
    manifest.update(exit_code=code, completed=bool(result and result.get("completed")))
    if result:
        manifest.update(
            model_gap_mm=result["settled_model_endpoint_error_mm"],
            duration_s=result["duration_s"],
            max_tracking_error_deg=result["max_tracking_error_deg"],
            control_timing=result["control_timing"],
            executed_planner_info=result["planner_info"],
        )
    write_json(folder / "trial.json", manifest)
    if Path("runs/live/rgb.png").exists():
        shutil.copyfile("runs/live/rgb.png", folder / "after.png")
    print(json.dumps(manifest), flush=True)
    if code or not result or not result.get("model_target_reached"):
        raise RuntimeError(f"Comparison trial incomplete; inspect {job.directory}")


if __name__ == "__main__":
    main()
