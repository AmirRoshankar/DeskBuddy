# FlowReach: one-step learned IK proposals

**Historical experiment:** the learned-IK option has been removed from the live
interface. Current Go uses numerical IK only. Retained code and results below
describe the earlier endpoint-proposal study. Flow is now used for separate
[trajectory models](trajectory-learning.md); see the [controller guide](../archive/controller-history.md)
for current UI behavior.

This obstacle-free experiment adapts one-step conditional action generation to
four unlocked joint-angle deltas. It uses the improved MeanFlow v-loss in
[Geng et al., Algorithm 1](https://arxiv.org/abs/2512.02012), with the predicted
instantaneous velocity as the JVP tangent and a detached JVP outcome. It is
inspired by [MP1](https://arxiv.org/abs/2507.10543) and the iMF action generator in
[ReactVLA](https://arxiv.org/abs/2606.14255). It does not reproduce their vision,
language, demonstration datasets, or full manipulation-policy architectures.

## What changes

Numerical mode solves IK from the current measured pose. Flow + IK first runs a
small conditional MLP once to propose final joint angles, then uses that proposal
as the initial guess for bounded numerical IK. The posture penalty still references
the actual start, and locked wrist roll remains unchanged. The same quintic timing,
speed bounds, path validation, serial feedback, Kalman estimation and PI tracking
apply to both modes. This does not increase motor speed.

The two refined solutions are usually effectively identical. In an offline audit
of 598 accepted learned-seed cases, the median largest angle difference was 0.001°.
One concrete held-out example had an elbow of −100.679° with numerical IK versus
−99.034° with iMF-seeded IK, while both modeled endpoints matched the same target.
This is a joint-solution example, not a validated hardware trajectory. See the
[saved examples](results/ik-solution-differences.json); the audit preceded removal of the
old raw-seed quality gates.

`flow_seed_strategy` accepts `direct` (default), `workspace_affine`, or `best`.
Affine seeding maps the current selection-box coordinates into the training box
before predicting angles; refinement always targets the **original** physical
point with unchanged robot dimensions. `best` ranks direct and affine candidates
by their current-model seed error and tries bounded refinement in that order.
This is a warm-start heuristic, not scale-invariant robot IK.

On 200 synthetic local-perturbation cases under the current configuration, direct
seeds had a median 2.17 mm model error versus 73.28 mm for affine seeds. Both
refined successfully in 199/200 cases; median proposal-plus-refinement times were
4.61 ms and 15.28 ms. Affine seeds were closer in none of these cases, so direct
remains the default. These are single-pass IK measurements, not path or hardware
validation; see the [remapping experiment](results/workspace-remapping.json).

The UI selector is optional; Numerical stays the default. A mode change clears the
previous preview. Planner diagnostics identify learned use versus numerical fallback.
Checkpoint fingerprints record training provenance. Camera, paper, bounds or tool
calibration changes do not block learned guesses: the current model and constraints
govern bounded IK and path validation. Extrapolation beyond the training input box
is explicitly reported; it is not a claim that the network generalizes accurately.

The model predicts four joint deltas from five current angles and target XYZ.
Its generative state has four dimensions; flow time is unrelated to physical time.
A fixed Gaussian seed makes a repeated preview reproducible. There is no claim
that this simple teacher dataset contains diverse valid routes to the same goal.

## Data, training and resources

All artifacts live together under ignored `runs/flowreach/`: dataset, metadata,
training log, MLP/iMF checkpoints, training summary and detailed benchmark.
No camera images or physical demonstrations are required. Synthetic pairs are
generated using the saved model, workspace and numerical IK. Endpoint clearance
is screened; complete trajectory validity is checked separately at deployment.

The first run generated 6,200 pairs: 5,000 training, 600 validation and 600 test.
Splits hold out entire 4 cm target cells. A fixed hash partitions cells; this tests
spatial generalization within the commissioned workspace, not arbitrary new robots.
The data are model-generated and therefore inherit model/calibration inaccuracies.

Both networks have three 192-wide hidden layers and were trained for 15,000 steps,
batch size 512, on an RTX 5070. Best checkpoints were selected using validation
error; the test split was used only afterward. MeanFlow has an initial ordinary
flow-matching warm-up and diagonal-time examples during training. No classifier-free
guidance or pretrained backbone is used. The supervised MLP is a strong comparison
for this largely deterministic task.

The artifact folder is capped at 10,000,000,000 bytes and checked before generation
and checkpoint writes. GPU allocator limits are set before allocation; the first
run peaked at 30.5 MB of PyTorch tensor allocation (excluding CUDA context/driver
overhead). Stored experiment artifacts total approximately 1.4 MB. Data generation
took about 95 s and training about 52 s on this machine.

## First held-out results

CPU inference uses loaded, warmed models and includes one iMF evaluation. It excludes
Python/PyTorch startup and checkpoint loading. Errors below are FK/model distances,
not observed physical tool accuracy.

| Method | Median time | Median model gap | Within joint bounds and 2 mm |
| --- | ---: | ---: | ---: |
| Numerical IK | 10.21 ms | <0.001 mm | 100% |
| Supervised MLP, raw | 0.11 ms | 2.22 mm | 42.2% |
| iMF, raw | 0.13 ms | 2.12 mm | 44.3% |
| MLP + numerical refinement | 3.22 ms | <0.001 mm | 100% |
| iMF + numerical refinement | 3.50 ms | <0.001 mm | 100% |

These numbers cover 600 held-out targets and describe endpoint solutions only.
For the first 100 cases, whole-path validation accepted 79 numerical paths and
80 Flow + IK paths; the others were rejected. Raw proposals outside joint ranges
are never executed. This is not a claim of 100% reachable or collision-free paths.

The learned warm-start reduced median IK computation by roughly 2.9×. The simple
MLP was comparably effective and slightly faster; this experiment does not establish
that iMF is superior. Neither result implies a 2.9× reduction in physical reach time.
See `flowreach-benchmark.json` for the public aggregate metrics and local
`runs/flowreach/benchmark.json` for per-case details.

## Refinement and deployment limits

Raw iMF missed the 2 mm model tolerance in over half the test cases. Numerical
refinement therefore happens before execution in this first version. The learned
model has not been used to command physical hardware. The existing feedback loop
continues correcting joint tracking during execution and settling.

The offline `coarse_refinement` experiment also tests the proposed ordering:
validated raw coarse segment, an ideal simulated encoder endpoint, then numerical
IK correction. Both stages passed validation in 72 of the first 100 held-out cases.
Among those cases, two stages added a median 1.95 s of trajectory time: the current
second segment has a 2 s minimum. This excludes additional settling and feedback
reacquisition overhead. That result supports refinement before execution for this
version. It does not rule out a future continuous or visually corrected approach. Physical endpoint
error also needs independent visual measurement before claiming improved accuracy.

Missing/incompatible checkpoints, nonfinite proposals, failed bounded refinement,
or rejected learned-seeded paths trigger numerical fallback. Finite proposals are
projected onto current joint bounds and refined, even outside the training range
or with large initial FK error. Raw guesses are never executed. Preview, progress
and completion logs report numerical fallback with its reason; successful learned
use stays quiet. Structured plan/report JSON retains the full solver diagnostics.
The original joint/workspace/table checks remain in force. No obstacle model or
full-body collision guarantee is added by learning.

## Reproduce offline

Install `.[flow,dev]` into the prepared environment, then run from the repo root:

```powershell
python -m deskbuddy.learned.data --config configs/local.json --folder runs/flowreach --train 5000 --validation 600 --test 600
python -m deskbuddy.learned.train --folder runs/flowreach --steps 15000 --batch 512 --width 192
python -m deskbuddy.learned.benchmark --folder runs/flowreach --samples 600
python -m deskbuddy.learned.coarse_refinement --folder runs/flowreach
```

Use `configs/local.example.json` for a portable example, or the commissioned local
config for the actual arm. A new training run overwrites this experiment's checkpoints.
There is no serial or camera access in these three commands. The operator's persistent
`runs/control/hardware_disabled` lock remains active until explicitly resumed;
offline training does not remove it or restore motor power.
