# Learning grasping and testing full-arm generalization

## Why the old demo looked better than the real arm

The old teacher generator (`trajectory/dataset.py`) moves a 2D tool proxy around
one or two objects, with starts mainly on the left and goals on the right. Its
condition vector has only two obstacle slots. It does not encode the initial
five-joint posture, obstacle height, table tilt, wrist orientation or gripper
opening. Its 18 mm tool inflation also differs from the live 35/45 mm arm capsules.
The live system groups extra boxes, lifts XY routes into joints, and checks much
more geometry. Passing an XY benchmark is not passing a real-arm benchmark.

Other effects compound this: a deterministic MLP can average different route
modes into an invalid middle route; calibration/model error and changing perception
affect execution; motor lag is absent from geometric simulation. More obstacles
alone cannot repair a representation that omits the arm's posture and height.

## Implemented shared full-arm experiment

`trajectory/joint_dataset.py` generates 32 × 5 joint trajectories from checked
configuration-space RRT teachers. It varies the initial and goal poses, 0–6 boxes,
thin walls, tall objects, raised boxes, table tilt/height, workspace limits and
wrist-lock conditions. All geometry is in the robot base frame. Robot dimensions
are never scaled with the playground. Axis-aligned boxes are still an approximation
to rotated or irregular objects; this is geometric simulation, not contact physics.

Entire scene families, including every query and teacher seed, stay in one split.
MLP, FM and iMF use one immutable NPZ artifact, common normalization, the same
minibatch row IDs, 10,000 updates, width 192, batch 256 and LR schedule. Their
objectives differ, and FM/iMF have more parameters because they consume a noisy
trajectory and time. Equal data/updates does not mean equal compute. MF is retired
from new training and the active UI; old artifacts remain historical records.

The first run contains 250 families / 2,000 scenarios / 4,000 teacher attempts.
3,782 passed the original full-arm checks; 139 were then rejected by the new
continuous body-to-table check, leaving **3,643 shared teachers**. Train/validation/
test rows are **2,898 / 373 / 372**. Failed attempts and rejected indices are retained.
The collision-body revalidation does not regenerate rejected teachers, so coverage
of difficult failures remains a limitation rather than an invented success label.

The shared artifact SHA-256 is
`c75e63a36153fb8d3edcde1b65569959f6031690221a244765ad6734850fec88`.
Every new checkpoint stores that fingerprint. Evaluation refuses mismatched models.

### Initial held-out results

See the [input/output, loss-equation and training audit](model-formulation-audit.md)
for exactly what MLP/FM/iMF receive and predict, what the iMF loss means, how the
checkpoints were selected, and the limitations exposed by this table.

80 endpoint-valid held-out scenarios across unseen families; 31 require leaving
the straight joint interpolation. Raw validity means the entire sampled arm route
passes joint, workspace, table, self and object checks **before any repair/fallback**.
Exact endpoints and locked joints are imposed for every method.

| Method | One candidate valid | Up to four valid | Blocked cases solved with four | CPU generation median, four |
|---|---:|---:|---:|---:|
| Straight joint interpolation | 49/80 | 49/80 | 0/31 | 0.02 ms |
| MLP, deterministic | 52/80 | 52/80 | 4/31 | 0.41 ms |
| FM, 8 evaluations | 18/80 | 33/80 | 1/31 | 0.94 ms |
| iMF, 1 evaluation | 48/80 | 49/80 | 0/31 | 0.53 ms |

The one-candidate column uses the first draw from the same four-candidate run;
it is not a separately timed run. Generation times exclude the much slower full-arm
validation, perception and motors. This modest sample does not establish a general
ranking. The new checkpoints are **research artifacts, not substituted into live Go**:
the blocked-case performance is inadequate. Training completed in about 54 seconds
on RTX 5070, but teacher generation and validation took substantially longer.

Next: increase blocked/narrow-passage coverage, add held-out live-layout replays,
use body/workspace losses and geometric guidance, and compare learned proposals
plus a common trajectory optimizer. Keep raw and repaired validity separate.
`trajectory/joint_learning.py` supplies differentiable FK/body costs but these were
not used in this first experiment. This follows the learned-prior plus cost-guidance
direction of [Motion Planning Diffusion (2024/2025)](https://arxiv.org/abs/2412.19948),
rather than assuming one-shot outputs satisfy collision constraints.

Reproduce from the repository folder:

```powershell
..\..\.venv\Scripts\python.exe -m deskbuddy.trajectory.joint_dataset --folder runs/deskbuddy/new-joint-data --families 250 --queries 8 --routes 2
..\..\.venv\Scripts\python.exe -m deskbuddy.trajectory.train --dataset runs/deskbuddy/joint-trajectories-v2 --output runs/deskbuddy/new-joint-models --steps 10000 --batch 256 --width 192
..\..\.venv\Scripts\python.exe -m deskbuddy.trajectory.joint_evaluate --dataset runs/deskbuddy/joint-trajectories-v2 --models runs/deskbuddy/new-joint-models --output runs/deskbuddy/new-joint-evaluation.json --limit 80 --candidates 4
```

New generation uses current collision-body checks. To reproduce the historical
two-stage dataset, revalidate an older source with `--revalidate SOURCE --folder NEW`.
Shared `runs/deskbuddy` storage remains capped at 10 GB. None of these CLIs opens a
camera, serial port or motor controller.

## What ML should learn for grasping

The [formulation audit](model-formulation-audit.md#paper-inspired-grasp-experiment)
also describes the requested GraspMF/Adaptive Mean Flow direction and why dummy
boxes provide geometry labels, not proof of physical key-grasp transfer.

A detector rectangle is not a grasp. The system must choose a pinchable point,
jaw yaw/opening, approach direction, contact height, closure timing, lift and success
evidence. Keys and loose chains make the distinction especially visible. Our current
open–descend–close–lift loop and retry offsets are hand-authored; it is not a learned
grasp policy. Numerical pose IK now enforces downward orientation, but that alone
cannot discover a good contact or compensate for bad tool/table calibration.

| Research / guide | Useful part | Fit here |
|---|---|---|
| [Transporter Networks, CoRL 2020](https://proceedings.mlr.press/v155/zeng21a.html) | Spatial visual action prediction | Learn a top-down grasp point/yaw rather than using a detection center. Its tabletop setting is relevant; it is not a ready-made keys policy. |
| [Diffusion Policy, RSS 2023 / IJRR 2024](https://diffusion-policy.cs.columbia.edu/) | Image-conditioned action sequences and receding-horizon execution | Learn approach, closure and lift jointly, then re-observe and predict again. |
| [Fast Generative Grasping via Lie Group-Constrained MeanFlow, 2026 preprint](https://arxiv.org/abs/2608.26076) | Fast distribution over grasp position and rotation | Closest MeanFlow grasp-synthesis connection. It is not iMF, not a complete contact policy, and its ACRONYM result does not establish performance on SO-101 keys/chains. |
| [Adaptive Mean Flow, 2026](https://arxiv.org/abs/2610.06089) | Fast closed-loop prediction with continuity to the previous plan | Relevant to re-planning smoothly as the gripper approaches, rather than executing a long blind action chunk. |
| [LeRobot ACT guide](https://huggingface.co/docs/lerobot/main/act), [record/train workflow](https://huggingface.co/docs/lerobot/main/il_robots) | Practical imitation-learning baseline and episode recording | Best first physical grasp baseline. Hugging Face recommends ACT as a starting policy; demonstration counts are task dependent. |
| [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) | Fine-tune an existing RGB/language/action model, including SO-101 examples | Useful later for multiple objects/tasks. It still needs demonstrations from this setup. |

Recommended next experiment: collect roughly 50 successful, varied real pickups,
plus labeled failures for a success verifier. Start with one rigid key/fob part,
fixed camera, known table and consistent approach; vary XY/yaw and initial arm pose.
Record synchronized RGB, all five joint angles, gripper position/commands, target
and timestamps. Split by episode and object arrangement, not adjacent video frames.
This target count is a starting budget, not a promised success threshold. The
[LeRobot data guidance](https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md)
also emphasizes starting constrained and expanding diversity after inspecting failures.

Train ACT as a practical reference. For the research comparison, use the same frozen
visual features and six-channel action chunks (five joints plus gripper) for MLP,
FM and iMF, with identical episodes/splits/budgets. Predict a short horizon, execute
only its beginning, re-observe, and re-plan. Keep numerical pose/geometry checking
at the output boundary. iMF can cheaply sample several grasp-and-lift alternatives;
its utility must be measured in closed-loop success/latency/retries, not an attractive
curve. No key-grasp policy has yet been trained: collision-free synthetic reaches
do not provide contact, friction, chain entanglement or successful-lift labels.
