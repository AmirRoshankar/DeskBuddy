# Reproduce the planner result

## One command

After installing Python 3.12+ and `pip install -e ".[flow]"`:

```bash
python scripts/build_planner_comparison.py --reproduce
```

The command downloads the checksummed v0.2.0 benchmark bundle when inputs are missing, evaluates the released MLP/FM/MF/iMF checkpoints on the same 500 test conditions, and recomputes the paired layout-family bootstrap. It does not open cameras or motors. Results go to `runs/deskbuddy/reproduction/`; published research records are preserved.

Without `--reproduce`, the script rebuilds the public comparison from recorded metrics. That mode does not rerun inference.

## Inputs and accounting

- Dataset: 8,000 training paths, 1,000 validation paths, and 500 evaluated held-out conditions from 20 test layout families.
- Direct routes are blocked in 245 of the 500 evaluated conditions.
- Model checkpoints: the original 20,000-step run, training seed 20261125.
- Evaluation noise: PyTorch seed 1729, shared across methods.
- One Torch CPU thread; timings include generation, validation and RRT fallback, excluding perception, joint-space lifting and actuation.
- Validity: every segment of the 32-point XY path is checked in physical coordinates before fallback.
- Four candidates: success means at least one raw candidate is valid.
- Bootstrap: 10,000 paired resamples over layout families, seed 20261125. The interval measures layout uncertainty conditional on these checkpoints, not training-seed uncertainty.

The bundle includes the exact synthetic dataset and all four original checkpoints, including the historical MF baseline. `models/manifest.json` records the archive and per-file SHA-256 digests. The installer rejects unsafe paths, checksum mismatches and differing local files.

Floating-point kernels can vary across PyTorch versions and platforms; timing depends on the CPU. Report any validity changes alongside the versions rather than editing them to match the reference.

## Training-seed study

Three fresh runs use seeds 20261125, 20261126 and 20261127, the same dataset fingerprint and the same MLP/FM/iMF training recipe. MF remains in the original benchmark but is retired from new training. This study is reported separately from the original five-row table.

```bash
python -m deskbuddy.trajectory.train --dataset runs/deskbuddy/trajectories-v1 --output runs/deskbuddy/trajectory-seeds/20261125 --seed 20261125 --steps 20000 --lr 0.0003 --diagonal-fraction 0.25 --adaptive-power 0.5 --imf-auxiliary-head
python -m deskbuddy.trajectory.evaluate --dataset runs/deskbuddy/trajectories-v1 --models runs/deskbuddy/trajectory-seeds/20261125 --methods mlp fm imf
```

Repeat with the other two seeds and fresh output folders. [Seed-study results](training-seeds.md) include per-seed scores and their spread. All teacher families retain their original split.

## Historical paths

The project began as Click-to-Reach, used FlowReach for learning experiments and FlowDesk for the desktop controller, then became DeskBuddy. New public commands and artifacts use `runs/deskbuddy/`. Older reports preserve `flowreach` as provenance, and the installer accepts the immutable v0.1.0 bundle paths. No robot dimensions are rescaled when changing workspace bounds.
