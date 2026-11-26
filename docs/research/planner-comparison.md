# Which planner is best so far?

These are results for 500 held-out synthetic conditions with fixed trained checkpoints and a fixed noise seed. Of those, 245 had an obstacle blocking the direct start-to-goal segment. They do not measure physical robot success.

**Raw valid** means the planner's own complete 32-point path passed workspace and continuous segment checks against inflated 2D obstacle geometry, before fallback. The checker tests the connecting segments, not just the waypoint dots. Start and goal are enforced explicitly. It does not test the SO-101 links, measured tip accuracy, acceleration limits or real depth uncertainty.

For four candidates, raw-valid rate is the fraction of conditions with **at least one valid candidate**. Rejected candidates are not executed. RRT fallback is counted separately.

## One candidate per condition

| Planner | Own path valid / 500 | Valid when direct route blocked | RRT fallback | Median total ms | P95 total ms |
|---|---:|---:|---:|---:|---:|
| RRT-Connect | 500/500 (100.0%) | 100.0% | 0.0% | 1.18 | 4.49 |
| MLP | 346/500 (69.2%) | 37.6% | 30.8% | 0.84 | 5.46 |
| FM (8 evaluations) | 367/500 (73.4%) | 47.8% | 26.6% | 0.85 | 3.83 |
| MF (1 evaluation) | 409/500 (81.8%) | 62.9% | 18.2% | 0.54 | 3.40 |
| iMF (1 evaluation) | 431/500 (86.2%) | 72.7% | 13.8% | 0.53 | 3.13 |

Total time includes generation, path validation and any RRT fallback. Timings are warmed CPU measurements with one Torch thread, excluding camera acquisition, perception, IK and actuation. A millisecond here does not mean a millisecond robot movement. All policies reached 100% final proxy validity after RRT fallback; that does not make their learned planners equally successful.

## Four candidates per condition

| Planner | At least one valid / 500 | Valid when direct route blocked | RRT fallback | Median total ms |
|---|---:|---:|---:|---:|
| FM (8 evaluations each) | 466/500 (93.2%) | 86.1% | 6.8% | 1.85 |
| MF (1 evaluation each) | 485/500 (97.0%) | 93.9% | 3.0% | 1.54 |
| iMF (1 evaluation each) | 488/500 (97.6%) | 95.1% | 2.4% | 1.60 |

The deterministic MLP does not gain alternative paths by being called four times with identical inputs. Four FM candidates cost 32 sample evaluations; four one-step MF/iMF candidates cost four. Batched calls reduce overhead, but all candidates still need checking.

## Practical ranking

- **Most reliable on these simple scenes:** RRT, with 500/500 own paths valid. It is the classical reference and recovery mechanism, not a neural model.
- **Strongest single-candidate neural result:** iMF, 431/500 versus MF's 409/500. Their total median times, 0.53 and 0.54 ms, are effectively too close to rank from this single timing pass.
- **Four-candidate MF versus iMF:** close. iMF succeeds on only three additional cases, while MF is slightly faster in this run. Neither is a convincing overall winner at this budget.
- **No universal method winner:** this is one training seed and a small family of synthetic tasks. iMF uses an auxiliary velocity network during training; inference generator capacity is matched, but training cost is higher.

A paired bootstrap over the 20 held-out obstacle-layout families estimates iMF minus MF at 4.4 percentage points for one candidate, with a 95% interval of 2.8 to 6.2. With four candidates, the estimate is 0.6 points, interval -0.4 to 1.6; this includes a tie or reversal. These intervals condition on the fixed checkpoints/noise seed and do not include training-seed variation.

## Other metrics and what is missing

The full report records generation and validation times separately, path length, minimum inflated-obstacle clearance, checkpoint selection score and NFE. Length/clearance summaries only include accepted model paths, so they cannot fairly crown a winner alone. Geometric second differences are recorded internally, but physical acceleration/jerk require time-parameterized joint trajectories. Goal endpoints are fixed by construction; measured physical positioning error and motor energy are not evaluated by this 2D experiment.

## Live status

All five planners can now receive the same live RGB-detected rectangular obstacles in a preview-only comparison. Object detection is shared, not a separate detector for each planner. YOLOE-26 Nano RGB instance masks define outlines; filtered depth estimates 3D boxes. The calibrated flat paper and modeled robot are excluded. The models have two obstacle slots; extra detections are conservatively merged into two rectangular regions, and every planner receives those same regions. Swapping objects updates detections and invalidates old previews.

The first camera-fed test with four detections grouped into two regions accepted 0/1 MLP candidates and 0/4 FM, MF and iMF candidates; RRT found a path. This is one script-selected target in an estimated scene, not a hardware success-rate benchmark. It exposes domain shift and possible perception/grouping conservatism rather than proving that the offline ranking transfers to the desk.

The main sidebar now selects RRT, MLP, FM, MF or iMF and one/four candidates for Go. Its route is lifted through numerical waypoint IK, time-parameterized, checked against joint/table/workspace and approximate arm-link/obstacle constraints, and executed without substituting a direct reach. Changed confirmed scene geometry stops execution. This executor has offline/fake-bus validation; physical Cartesian-route trials have not yet been run. Compare planners remains preview-only. Direct is a separate endpoint-only baseline blocked by detected objects.

[Full offline metrics and plots](trajectory-learning.md) · [How the Go button works](../archive/controller-history.md)
