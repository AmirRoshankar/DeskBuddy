# References and learning resources

These resources informed the project. The implementation adapts selected ideas to an SO-101; it does not reproduce every method below.

| Resource | What I used or studied |
| --- | --- |
| [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) — Geng et al., 2025 | Interval-average velocity and one-evaluation generation; MF baseline. |
| [Improved Mean Flows](https://arxiv.org/abs/2512.02012) — Geng et al., 2025 | Predicted-velocity JVP training and the auxiliary velocity head used by the iMF adaptation. |
| [Adaptive Mean Flow for Responsive Closed-Loop Robot Control](https://arxiv.org/abs/2610.06089), 2026 | Motivation for refreshing actions when observations change; an extension direction rather than an implemented reproduction. |
| [Fast Generative Grasping via Lie Group-Constrained MeanFlow](https://arxiv.org/abs/2608.26076), 2026 | Grasp-pose generation on rigid-motion geometry; background for the grasp experiments. |
| [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763) — Bewley et al., 2016 | Kalman prediction and assignment as a tracking baseline; background for stable detections. |
| [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465) — Wang et al., 2025 | Open-vocabulary object detection and segmentation. |
| [Open Motion Planning Library](https://ompl.kavrakilab.org/) | Sampling-based planning reference. DeskBuddy implements its own RRT routines; OMPL is not a runtime dependency. |
| [LeRobot](https://github.com/huggingface/lerobot) | SO-101 motor interface, commissioning and robot-learning tooling. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) — Todorov, Erez and Tassa, 2012 | Gravity, contact and grasp-retention experiments. |
| [MIT 6.8210: Underactuated Robotics](https://underactuated.csail.mit.edu/) — Russ Tedrake | Dynamics, control, optimization and model-based reasoning. |

For the exact equations, inputs and losses implemented here, see the [model formulation audit](model-formulation-audit.md).
