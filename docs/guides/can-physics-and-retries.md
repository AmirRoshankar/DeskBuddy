# Can pickup evidence, retries and contact-physics experiment

This experiment tests whether training on simulated contact outcomes improves can pickup and transfer. For implementation details, see the [development guide](../MAINTAINER.md).

## Post-lift behavior

The worker now compares fresh RGB can-body appearance against two hypotheses: it stayed on the table, or it moved with the gripper. Raw JPEGs are published by the viewer; annotated previews, stale AABBs, gripper closure and depth are not pickup evidence. Two distinct frames must agree, with no contradictory visible-camera votes.

Confirmed pickup permits transfer and release. Confirmed miss replans a different approach, at most **five total attempts**. Unknown evidence retains hold without release/retry. Every new approach is preflighted from measured joints; iMF avoids tried radial directions and the numerical baseline varies closing yaw. Five misses never proceed to the discard phase. Drop success itself remains unverified.

This is a visual motion estimate, not contact/force sensing. Template matching may be uncertain under occlusion, low texture, can spin or a bumped camera. The physical can-to-discard demonstration is shown in the [README](../../README.md#demo). This report evaluates the separate simulated grasp checkpoints.

## Physics is different from geometric pose validity

MuJoCo 4.0.1 simulates gravity -9.81 m/s², a dynamic rigid can/table, finite-force parallel fingers and frictional contact. The dynamic hand tracks a mocap target through a weld. The full SO-101, actual actuator gains/fingers, aluminum deformation and liquid slosh are not simulated.

Explicit toy assumptions: 6 mm lid recess, 1 mm lip radius, 24 capsule ring segments, 3×8×24 mm finger boxes, 4 N/finger actuator ceiling, mass 15–400 g and friction 0.35–1.1. Contact shape and inertia are approximate, not measured hardware. Grasp success requires retained can-bottom clearance at lift/transfer, then settling in a simulated discard zone; counting a raised body origin alone would falsely count a can tipped onto its side.

| Experiment | Trials | Complete simulated successes |
| --- | ---: | ---: |
| Broad rim perturbations | 20,000 | 67 (0.335%) |
| Pilot-informed deeper contacts | 12,000 | 227 (1.892%) |

The changed proposal distribution means the two rows are **not a matched model comparison**. Both retain every failure in `all-rollouts.npz`. After pilot exclusion, 285 successful examples from 189 independent can IDs train a separate iMF checkpoint. Overlapping broad/focused object IDs stay in the same split. Pilot IDs 0–39 are excluded. Training: width 128, AdamW 0.0003, batch 256, 10,000 steps, CUDA 26.5 s.

On **246 matched held-out cans**, one proposal and one NFE per model:

| iMF checkpoint | Simulated pickup | Complete pick/transfer/drop |
| --- | ---: | ---: |
| Original geometric labels | 19/246 (7.72%) | 3/246 (1.22%) |
| Physics-success labels | 44/246 (17.89%) | 6/246 (2.44%) |

Low counts and low success do not establish reliable superiority or real transfer. The physics checkpoint remains an **offline experiment**, not silently substituted into the live action. The original geometric checkpoint remains the current live source. A better grasp pose alone may not solve retention; contact geometry, jaw force, carried-object orientation and closed-loop grasp control need further work.

Artifacts: `runs/deskbuddy/can-rim-physics-v1/`, `can-rim-physics-v2/`, `can-rim-physics-training-v1/`, and `can-rim-grasps-physics-v1/`. Exact per-model held-out outcomes are in the last folder's `physics-evaluation.json`. Source: `learned/can_physics.py`, `can_physics_evaluate.py`, `grasp_lab.py --physics-dataset`. The table above measures the simulated checkpoints, not a physical success rate.
