# Optional models

The CPU numerical demo and geometric planning work without checkpoints. Learned path/grasp inference needs the project's separately distributed **synthetic-data checkpoints**:

```bash
python scripts/download_models.py
```

The script verifies the release ZIP and each checkpoint against `models/manifest.json`, rejects unsafe paths and preserves differing local weights unless `--replace` is explicitly supplied. It does not access hardware.

The v0.2.0 bundle includes the original XY trajectory MLP, FM, MF and iMF checkpoints, the exact synthetic benchmark dataset, and the existing grasp checkpoints. All artifact paths use `runs/deskbuddy/`. The immutable v0.1.0 release remains available for historical experiments.

The bundle contains project-authored weights and synthetic data. It excludes camera images, machine calibration and third-party detector weights. For a complete inference rerun, see [reproducibility](../research/reproducibility.md).

## Reproduce instead of downloading

The training code lives in `deskbuddy.trajectory` and `deskbuddy.learned`. Each module's `--help` lists exact arguments. For example:

```bash
python -m deskbuddy.learned.grasp_lab --profile can_rim --objects 2500 --steps 10000 --folder runs/deskbuddy/new-geometric-rim
python -m deskbuddy.learned.can_physics --folder runs/deskbuddy/new-physics --objects 1500 --attempts 8 --focused
python -m deskbuddy.learned.grasp_lab --profile can_rim --steps 10000 --physics-dataset runs/deskbuddy/new-physics/dataset.npz --folder runs/deskbuddy/new-physics-model
```

Use fresh output directories. Physics training refuses insufficient successful examples rather than training on an empty set. Preserve object IDs across splits and retain failed rollouts in evaluation accounting. Full experiment details and limitations are in the research guides.

## External detector

Live can perception separately expects `runs/deskbuddy/perception/yoloe-26s-seg-pf.pt` and an installed `perception` extra. Obtain it through the pinned Ultralytics asset downloader or its [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/), under the [upstream terms](https://www.ultralytics.com/license). Those weights are not part of DeskBuddy's checkpoint release.
