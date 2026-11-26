# Optional models

The CPU numerical demo and geometric planning work without checkpoints. Learned path/grasp inference needs the project's separately distributed **synthetic-data checkpoints**:

```bash
python scripts/download_models.py
```

The script verifies the release ZIP and each checkpoint against `models/manifest.json`, rejects unsafe paths and preserves differing local weights unless `--replace` is explicitly supplied. It does not access hardware.

The v0.1.0 bundle contains nine project-authored PyTorch state dictionaries:

- Current XY trajectory MLP, FM and iMF in `runs/flowreach/trajectories-v1/models-v3/`.
- Geometric rim iMF in `runs/flowreach/can-rim-grasps-v1/` (current live grasp proposal source).
- Experimental physics-filtered rim iMF in `runs/flowreach/can-rim-grasps-physics-v1/` (not automatically selected).
- Earlier body-grasp MLP, FM, iMF and GraspMF in `runs/flowreach/can-grasps-v1/`.

The bundle does **not** contain raw images, hardware calibration, encoder-specific IK checkpoints, third-party weights or a claim of real pickup success. The geometric and physics model results are different metrics; see [physics and retries](can-physics-and-retries.md).

## Reproduce instead of downloading

The training code lives in `deskbuddy.trajectory` and `deskbuddy.learned`. Each module's `--help` lists exact arguments. For example:

```bash
python -m deskbuddy.learned.grasp_lab --profile can_rim --objects 2500 --steps 10000 --folder runs/flowreach/new-geometric-rim
python -m deskbuddy.learned.can_physics --folder runs/flowreach/new-physics --objects 1500 --attempts 8 --focused
python -m deskbuddy.learned.grasp_lab --profile can_rim --steps 10000 --physics-dataset runs/flowreach/new-physics/dataset.npz --folder runs/flowreach/new-physics-model
```

Use fresh output directories. Physics training refuses insufficient successful examples rather than training on an empty set. Preserve object IDs across splits and retain failed rollouts in evaluation accounting. Full experiment details and limitations are in the research guides.

## External detector

Live can perception separately expects `runs/flowreach/perception/yoloe-26s-seg-pf.pt` and an installed `perception` extra. Obtain it through the pinned Ultralytics asset downloader or its [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/), under the [upstream terms](https://www.ultralytics.com/license). Those weights are not part of DeskBuddy's checkpoint release.
