# Maintainer handoff

DeskBuddy is an SO-101 robotics-learning workbench. The code was consolidated from a longer Click-to-Reach/FlowDesk experiment; historical research names remain in artifact folders. The Python package and CLI are now **`deskbuddy`**.

## Read first

[Project overview](../README.md), [current controller guide](guides/controller.md), [installation](guides/installation.md), [architecture](guides/architecture.md), [model formulations](research/model-formulation-audit.md), and [physics/retry results](guides/can-physics-and-retries.md).

Machine-specific ports, transforms, private images and the full operational handoff are deliberately not public. On the original development machine, they are in ignored `configs/local.json`, `runs/`, and `local/FLOWDESK_HANDOFF.md`. Read the local handoff and obtain the operator's current authorization before hardware work. A repository clone or a passing test does not authorize motion.

## Architecture map

| Responsibility | Modules under `src/deskbuddy/` |
| --- | --- |
| Entry points/operator events | `cli.py`, `desktop_launcher.py`, `gui.py` |
| UI/input/3D | `ui/layout.py`, `theme.py`, `scene_inspector.py`, `viewport.py`, `hover.py` |
| Kinematics and calibration | `kinematics.py`, `planning.py`, `joint_limits.py`, `test_bed.py`, `multiview.py`, calibration modules |
| Acquisition | `camera.py`, `webcam.py`, `latest_camera.py` |
| Can perception | `trajectory/yolo_perception.py`, `rgb_can_perception.py`, `multiview_evidence.py` |
| Stability | `trajectory/object_tracker.py`, `scene_stability.py`, `scene_monitor.py` |
| Trajectory proposals | `trajectory/live_preview.py`, `learning.py`, `train.py`, `dataset.py` |
| Full-arm planning/checks | `trajectory/execution.py`, `joint_planner.py`, `self_collision.py`, `table_collision.py` |
| Ownership and execution | `go_control.py`, `hold_control.py`, `hardware_interlock.py`, `execute_reach.py`, `feedback_loop.py` |
| Can task | `can_discard.py`, `can_pickup.py`, `learned/grasp_proposals.py` |
| Grasp training/simulation | `learned/grasp_lab.py`, `can_physics.py`, `can_physics_evaluate.py` |

Keep numerical IK, XY learned paths and learned grasp poses conceptually separate. The live default endpoint solver is numerical. Whole-arm lifting/validation is required after a valid XY proposal. The physics grasp model is not a learned full-arm dynamics/contact controller.

## Runtime invariants

- Startup, camera reconnection, target selection and offline tests never enable torque or start motion.
- Stop retains hold; Disable releases all six servos. Preserve supported-arm and hands-clear checks.
- `runs/control/pause_robot` grants one worker serial ownership; never delete it to force access. The viewer releases its passive connection while a worker runs.
- `runs/live/stop` requests viewer exit without torque changes. Use `runs/live/process_id.txt` for the actual Python PID, not the launcher shell PID.
- Atomic snapshots are in `runs/live/state.json`. Fresh raw pickup images have their own path/timestamp fields; annotated `rgb.png` is not pickup evidence.
- Interpret encoder angles with configured signs/zero offsets. Use measured start pose, not a stale hold goal or renderer preview.
- Model/geometry residual, servo tracking error and physical tip error are distinct. Do not fix a false calibration error by silently broadening joint ranges.
- Keep requested loop frequency separate from achieved timing, command step and degrees/second.
- A scene revision alone does not imply a collision. Recheck remaining geometry and replan from measured joints when necessary; Stop prevents continuation.
- Post-lift ambiguity never triggers release or a blind retry. Only a confirmed miss allows another checked approach, at most five total attempts.

## Current evidence and limits

The numerical reaching/controller baseline has had small hardware trials. The newer rim-grasp and visual retry path have offline tests but no established real pickup-success measurement. Geometric rim validity was89.8%; physics simulation complete-task rates were3/246 versus6/246 for geometric/physics-trained iMF. These are not comparable kinds of success and do not establish reliable superiority.

The MuJoCo model includes gravity and rigid frictional contacts with a finite-force gripper. It does not reproduce the entire SO-101, measured fingers/lid, aluminum deformation or liquid slosh. `power_estimate.py` is an assumed quasi-static energy model, not measured electrical telemetry.

## Development and publication

Use [the documented checks](../README.md#development). Tests and the synthetic preview must work without connected hardware. Keep model assets in the checksummed release bundle and external detector weights under their own terms. Never commit local recordings/calibration or a machine-specific handoff. Historical documents live in `docs/archive/`; current guides override them.

Next useful work: validate real jaw geometry/force and RGB pickup evidence under operator control, improve retention/contact strategy, collect outcome labels, and compare matched MLP/FM/iMF data on whole-arm constraints and end-to-end planning latency. A larger network alone is not demonstrated to solve the sim-to-real gap.
