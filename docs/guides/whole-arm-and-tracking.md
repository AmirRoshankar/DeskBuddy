# Tracking, full-arm planning and key pickup

## What publications do, and what changed here

[SORT (Bewley et al.)](https://arxiv.org/abs/1602.00763) uses Kalman state estimation
and Hungarian assignment to connect noisy detections across frames. Our desk
tracker uses those building blocks on world center/size and image rectangles,
with a low-motion prior, innovation gating, bounded updates, stable IDs and short
occlusion retention. It is inspired by SORT, not a reproduction or benchmark claim.
A 4 mm center / 3 mm size / 3 pixel update limit damps accepted measurements.
Three consistent outlier observations reinitialize a relocated object rather than
inventing a slowly sweeping obstacle between its old and new locations. The
existing publication deadband keeps tiny filtered changes out of the planner.
Raw detections remain in snapshot diagnostics for comparison.

YOLOE-26 Medium prompt-free segmentation replaces Nano. Official weights:
[yoloe-26m-seg-pf.pt](https://docs.ultralytics.com/models/yoloe/), approximately
38 MB / 18.4 million parameters. A larger detector is not proof of accurate keys
recognition; class/confidence and calibrated paper exclusion still matter.
The prompt-free model mislabeled the actual key cluster as tools. A separate
Small visual-prompt detector now uses one boxed reference of the user's actual
keys, a supported [YOLOE visual-prompt mode](https://docs.ultralytics.com/models/yoloe/).
This is one-shot instance matching, not a newly trained general key detector.
Two consecutive matching detections are required before automatic targeting.
The thickest interior mask pixel supplies the pickup point. The reference and
weights stay private under `runs/flowreach/perception/`; nothing is bundled or
published. On fresh camera frames the key cluster was confirmed three times,
most recently around 60% confidence. Physical pickup has not been tested.

[OMPL](https://ompl.kavrakilab.org/stateValidation.html) separates configuration
validity from motion-edge validity. The live RRT now searches **five joint angles**
(with configured joints locked), not a 2D tip polyline. Every sampled state and
interpolated edge checks all approximated links against objects, table and
workspace. Non-neighboring links are also checked for self-collision. Adjacent
links sharing joints and components of the same wrist/gripper body are excluded
from self-pair checks. [cuRobo](https://nvlabs.github.io/curobo/latest/reference/self_collision.html)
uses robot collision spheres and configured self-pair exclusions; we use a small
CPU capsule model, not cuRobo or a detailed mesh model. Search edges are sampled
at 1 degree, with denser object screening of the final reference. Discretization
and imperfect geometry do not constitute a continuous physical collision guarantee.

The recorded desk regression went from rejected tip-only plans to a joint route
with about 13.6 seconds planned motion, versus roughly 34 seconds for the older
XY detour, in one offline replay. This is not a live performance benchmark.

## Making learned trajectories whole-arm aware

[Motion Planning Diffusion (Carvalho et al., 2024/2025)](https://arxiv.org/abs/2412.19948)
learns trajectory priors and applies planning-cost guidance; its manipulator
experiments go beyond end-effector-only geometry. The applicable change here is
joint-space training data plus robot-body collision costs, rather than increasing
an XY tip margin. The current checkpoints learned 32 XY points and cannot acquire
whole-arm reasoning just by changing a runtime threshold.

Implemented preparation:

- `trajectory/joint_planner.py`: full-arm teacher/search with joint and edge checks.
- `trajectory/joint_learning.py`: differentiable FK matching the runtime URDF;
  sampled link/obstacle and non-neighbor self-collision penalties with gradients
  through the joints.
- `trajectory/learning.py`: Field supports configurable condition/state dimensions;
  defaults preserve old checkpoint compatibility. A 32-point, 5-joint trajectory
  has 150 interior outputs instead of 60. Conditioning can use start/goal joint
  poses and 3D obstacle boxes; the endpoint IK stays numerical.

Next training run should generate diverse joint-RRT teachers, split by scene,
train MLP/FM/MF/iMF with matched data and budgets, add body losses or guidance,
and evaluate full-arm validity on unseen layouts. FK/body losses guide learning;
independent geometric checking remains necessary. **No new full-arm checkpoint
has been trained or substituted in the UI.** Existing learned routes are still
checked; a required joint-space RRT repair/fallback is labeled in logs and metadata.

## Key pickup UI and implementation

**Click & grab**: select the keys/chain on the paper, then press Go. **Grab keys**:
use a confident YOLO key/keychain detection on the calibrated test area to select
and start automatically. If no confident keys are detected, it refuses to guess.
Keys are thin grasp targets; their semantic detections are kept separately from
raised obstacle boxes. The target height comes from the known paper plane.

The sequence opens the gripper, follows the approach preview, descends to a
nominal point 2 mm above the table-clearance plane (about 5 mm above physical A4),
closes, lifts to the chosen hover height and verifies. It allows one initial
attempt plus five retries with small 2 mm pickup offsets. Contact mode allows only
the jaw/tip reference points near the table and extends the contact target's lower
workspace bound if needed; other links retain table clearance. Descent/lift are
preflighted before gripper torque enable. Encoder goals are checked before
enabling only the gripper; arm torque is not enabled by this routine.

Positive key observations near the lifted gripper, displaced from the original
pickup image location, are required across two distinct detector updates. Depth
above the paper can support this; clear RGB displacement can also support it when
depth is missing. Disappearance from the table alone is not success. This is an
experimental heuristic, not a trained grasp verifier like the object-persistence
representations studied in [Grasp2Vec](https://arxiv.org/abs/1811.06964).

Stop/Disable cancels pending automatic launches and active retries, retaining arm
and gripper hold. An unconfirmed final attempt remains unconfirmed; it does not
report success. Gripper motion is ramped and monitored for following error and
temperature. No force sensor or chain-entanglement model is present. The grasp
state machine and gripper adapter have fake-bus tests; physical key pickup has
not been commissioned or measured. Camera/arm calibration and jaw height may
need adjustment before reliable tabletop pickup is possible.


## Current camera-only validation

A 16-frame stationary capture with Small + Kalman tracking reduced world-center
standard deviation from [0.13, 0.54, 1.32] mm to [0.07, 0.20, 0.72] mm in X/Y/Z.
This short scene-specific measurement is not an accuracy benchmark or a guarantee
for all layouts. Private raw/filtered traces are in
`runs/flowreach/perception/small-tracker-validation.json`.


To create a new local key reference after swapping keys or changing the view,
use a saved RGB image and a box around only the keys:

```powershell
..\..\.venv\Scripts\python.exe -m deskbuddy.key_reference --image path\to\reference.png --box X1 Y1 X2 Y2
```

Run from the repository folder. This CLI uses no hardware and does not move the
robot. The promptable Small checkpoint must already be in the local cache; the
UI never silently downloads or installs anything. Cached visual embeddings avoid
loading a text encoder during runtime. Reload DeskBuddy after changing the reference.
