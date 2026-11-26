# iMF can-rim grasp experiment

The UI action **iMF rim grasp** detects the upright can from paired RGB views, samples object-relative lip poses, selects a feasible approach, pinches the rim, lifts, transfers to the calibrated discard zone, and opens. Only the user action starts execution. Depth is not used in can detection, localization or height selection.

## What is learned

The network predicts the grasp pose and opening; numerical IK and the whole-arm planner construct the four motion phases. This is a conditional one-step grasp proposal model, not an end-to-end learned manipulation or force-control policy.

Input: five context values, `[diameter, diameter, height] / 0.05`, plus cosine and sine of the radial approach angle. At runtime the system samples 32 approach directions automatically; the operator need not choose one.

Output: eight values, object-relative XYZ / 0.05, scalar-first quaternion, jaw opening / 0.05. The closing axis is radial, tool direction downward. Normalize the predicted quaternion before use. Pose offsets translate through the detected can's base center into robot coordinates; robot dimensions and joint limits do not scale.

The iMF field has width 128, context dimension 5 and state dimension 8. Training uses the existing `trajectory.learning.loss` iMF objective: data-to-noise interpolation, instantaneous model velocity as JVP tangent, detached derivative correction, adaptive squared-error weighting, 500-step diagonal warmup then 50% diagonal samples. Sampling uses one field evaluation from Gaussian noise. This Euclidean quaternion representation is normalized after sampling; it is not a manifold-flow implementation.

## Toy labels and training

Generate 2,500 synthetic cans, four directions each: **10,000 examples**. Diameter 60–74 mm, height 115–130 mm. Rim-center targets are at radius diameter/2 with +/-0.5 mm radial variation, 2 mm below the can top. Approach opening 7–9 mm. Labels express geometry, not successful contacts.

Split by can identity: 8,000 training examples, 1,000 test examples; 1,000 reserved examples unused. AdamW, learning rate 0.0003, batch 256, gradient norm cap 5, 10,000 steps. Training took 46.5 seconds on CUDA. Checkpoint and dataset: `runs/flowreach/can-rim-grasps-v1/`.

Reproduce into a new folder:

```powershell
python -m deskbuddy.learned.grasp_lab --profile can_rim --objects 2500 --steps 10000 --folder runs/flowreach/can-rim-grasps-v2
python -m deskbuddy.learned.grasp_lab --evaluate --folder runs/flowreach/can-rim-grasps-v2
```

## Held-out results

One draw: **89.8%** passes the procedural checks. Any of four draws: **97.6%**. First training-report draw: mean radial error 0.57 mm, mean height error 0.75 mm. Checks require <2 mm radial and height error, downward orientation within 12 degrees, radial closing and requested approach agreement within 15 degrees, and 5–12 mm opening. These are synthetic pose metrics, not pickup success rates.

Live geometry assumes **66 mm diameter and 122.22 mm height**. The nominal closing gap is 1.5 mm. Numerical IK may tilt the approach up to 35 degrees when strict vertical is unreachable; that projection is separate from the raw learned-pose metric. Position tolerance remains 2 mm. Entire approach, descent, lift and transfer are preflighted. The held cylinder is additionally checked against the table during lift/transfer using a nominal rigid attachment.

## Limits and next physical measurement

Jaw percentage-to-opening mapping is nominal URDF geometry. No friction, lip compliance, actuator force, can deformation, mass or contact dynamics are learned. A closed lid may prevent one jaw entering the rim, depending on fingertip shape and approach; the data does not establish that the robot can pinch this particular can. Camera extrinsics and can-center estimates remain important. Measure actual jaw opening and repeatability, then record real pickup outcomes before claiming sim-to-real grasp success. Completion logs explicitly leave pickup unverified.
