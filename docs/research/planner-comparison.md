# Planner comparison

500 held-out planning problems from 20 layout families; 245 have a blocked direct route.

| Planner | Own path valid | Blocked-route valid | Median total ms |
| --- | ---: | ---: | ---: |
| RRT-Connect | 100.0% | 100.0% | 1.18 |
| MLP | 69.2% | 37.6% | 0.84 |
| FM, 8 evaluations | 73.4% | 47.8% | 0.85 |
| MF, 1 evaluation | 81.8% | 62.9% | 0.54 |
| iMF, 1 evaluation | 86.2% | 72.7% | 0.53 |

With four candidates, iMF has at least one valid path in 97.6% of problems (95.1% with blocked direct routes).

The paired layout-family bootstrap estimates iMF minus MF at 4.4 percentage points, 95% interval [2.8, 6.2]. This interval conditions on the fixed training checkpoints and sampling seed; training-seed variation is reported separately.

## What valid means

Raw validity checks every segment of the complete 32-point XY path against the workspace and inflated obstacle geometry, before RRT fallback. It describes a tool proxy, not full-arm feasibility. Four-candidate validity means at least one candidate passes the same checks.

Timings are warmed CPU measurements with one Torch thread. Total time includes generation, validation and any RRT fallback; camera processing, joint-space lifting and motor execution are excluded.

## Interpretation

iMF is the strongest one-candidate neural result in this run. RRT solves all proxy problems. The harder full-arm experiment exposes a different limit: iMF solves 0/31 blocked cases. The next step is whole-body conditioning and online replanning, not assuming the proxy ranking transfers unchanged.

[Full model and training setup](trajectory-learning.md) · [Full-arm experiment](learning-grasps-and-generalization.md) · [Reproduction guide](reproducibility.md)
