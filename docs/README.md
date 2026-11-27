# Documentation

## Start here

- [Installation and offline demo](guides/installation.md)
- [Controller and camera controls](guides/controller.md)
- [Hardware and calibration](guides/hardware.md)
- [Architecture](guides/architecture.md)
- [Optional model checkpoints](guides/models.md)
- [Maintainer handoff](MAINTAINER.md)

## Learning and experiments

- [Model formulation audit](research/model-formulation-audit.md): inputs, outputs, losses and distinctions between IK, trajectory and grasp tasks.
- [Trajectory learning](research/trajectory-learning.md) and [planner comparison](research/planner-comparison.md).
- [Rim-grasp training](guides/can-rim-grasp-training.md).
- [Physics, gravity and pickup retries](guides/can-physics-and-retries.md).
- [Harder whole-arm/grasp experiments](research/learning-grasps-and-generalization.md).
- [Whole-arm tracking and validation](guides/whole-arm-and-tracking.md).
- [Communication measurements](reference/communication_timing.md) and [accuracy investigation](reference/accuracy-investigation.md).

`research/results/` contains compact measured JSON summaries. `assets/` contains the public success video, its animated preview, UI fixtures and plots. Reports distinguish geometric validity, simulation results and physical measurements; do not compare them as the same metric.

## Historical material

`archive/` preserves older operator instructions and plans for context. They are explicitly historical and may contradict later behavior. Use the current guides and code for operation. In particular, **Disable now releases torque**, and current can perception is RGB-only with YOLOE Small. Earlier research names FlowReach/FlowDesk describe stages of this project, now named DeskBuddy.
