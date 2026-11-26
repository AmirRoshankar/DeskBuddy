# Small hardware comparison — 25 November 2026

1 cm vertical reaches, iMF up, MLP return, MLP up, attempted iMF return; each accepted proposal refined with numerical IK before motion.

| Trial | Model gap (mm) | Peak tracking error (deg) | Elapsed (s) | Loop (Hz) | Joint settling met | Planned energy estimate (J) |
|---|---:|---:|---:|---:|:---:|---:|
| imf-up | 1.86 | 0.84 | 6.01 | 249.5 | False | 2.20 |
| mlp-return | 1.19 | 1.01 | 6.02 | 249.2 | False | 2.19 |
| mlp-up | 0.95 | 1.24 | 3.06 | 249.6 | True | 2.20 |

All three executed reaches completed within the configured model tolerance, with zero read/write failures or goal mismatches. The final iMF return was rejected before motor movement because its raw prediction exceeded calibrated joint bounds.

- Endpoint gaps are FK estimates from joint feedback, not independent physical tip measurements.
- Only three executed reaches; starts differ slightly, and the planned counterbalanced comparison was incomplete. No general model ranking is justified.
- The UI green marker remained the previous clicked target, not the external comparison target.
- Power is a quasi-static scenario for the 2-second planned path only, excluding settling and subsequent holding; electrical current was not measured. See flowreach-analysis.md for assumptions.
- First iMF and MLP-return reached model tolerance but timed out on the stricter joint settling criterion; MLP-up met both.

The last verified state was hold enabled at the raised position, with gripper torque off. After testing, viewer position reads began timing out on all motor IDs; current hold could no longer be verified. No torque-release command was sent.

[Detailed results](results/flowreach-hardware.json). Raw trial logs and camera snapshots remain local under `runs/hardware-comparison/2026-11-25/`.
