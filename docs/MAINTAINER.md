# Development guide

DeskBuddy has three layers: an offline robotics core, a desktop interface, and hardware adapters. Keep geometry, planning and learning usable without connected cameras or motors.

## Where to start

- `kinematics.py`, `planning.py`: FK/IK, trajectory validation and timing.
- `trajectory/`: classical and learned planners, scene tracking and collision checks.
- `gui.py`, `ui/`: operator actions, the shared 3D view and camera displays.
- `feedback_loop.py`, `execute_reach.py`: encoder feedback and motion execution.
- `can_discard.py`, `can_pickup.py`: grasp, verification, transfer and retry logic.
- `point_tracking.py`, `local_visual_servo.py`: selected-feature tracking and local image-based correction.
- `learned/`: grasp models, training and MuJoCo experiments.

The research benchmark models XY paths. The full-arm experiment models joint trajectories. Keep their datasets, validity definitions and metrics separate.

## Checks

```bash
python -m pip install -e ".[dev,flow,simulation]" opencv-python
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m deskbuddy demo --headless --output runs/offline
```

CI runs offline checks on Ubuntu and Windows. Windows is the primary hardware UI platform. Linux supports the offline experiments and desktop preview with Tk and a display; CI uses Xvfb.

## Artifacts

Use `runs/deskbuddy/` for new datasets, checkpoints and reports. Download the public benchmark bundle with `python scripts/download_models.py`; files are checksum-verified. `configs/local.json` holds machine-specific settings and stays out of Git. Preserve the dataset fingerprint and training seed in every experiment.

For the main result, run `python scripts/build_planner_comparison.py --reproduce`. See [reproducibility](research/reproducibility.md) for data, seeds and metric definitions.

## Hardware

Clamp the base, keep the workspace clear and stay beside the arm during a run. Support the arm before pressing Disable: it releases motor torque. Start with small motions when changing calibration or control code.

I use AI coding tools for implementation, debugging and test development. The model definitions, experiment inputs and evaluation scripts are included so the results can be inspected and rerun.
