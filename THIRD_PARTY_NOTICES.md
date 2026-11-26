# Third-party model

`src/deskbuddy/models/so101_new_calib.urdf` is a copy (with normalized line endings; geometry unchanged) from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
commit `385e8d7c68e24945df6c60d9bd68837a4b7411ae`,
path `Simulation/SO101/so101_new_calib.urdf`.

Upstream distributes it under Apache License 2.0; the license is included in `LICENSE`.
The model was generated with onshape-to-robot. Upstream contributors retain their rights.
This project also uses Apache License 2.0 for its original code.

The URDF and compressed convex-hull vertices of its collision meshes are bundled.
`collision_hulls.json` records the same upstream commit and source STL SHA-256
hashes. Convex hull support is exact for distance to a plane; these hulls are used
for physical table clearance and the 3D viewer. Object collision checks still use
arm capsules. This is not a contact or dynamics simulator.

## Optional RGB detector

The local optional perception environment uses Ultralytics 8.4.219 and
YOLOE-26 Small prompt-free segmentation weights, distributed by Ultralytics
under AGPL-3.0 or its commercial license. They are not bundled with this
repository. See https://www.ultralytics.com/license and
https://docs.ultralytics.com/models/yoloe/ for upstream terms and model details.

The optional prompt-encoding experiment used the Ultralytics CLIP fork at
commit `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40` (MIT) and an official MobileCLIP2
text-encoder asset. These caches are local, not repository assets. Runtime visual
key matching loads cached visual embeddings and does not download text encoders.


## Optional contact simulator

The gravity/contact experiments use MuJoCo 4.0.1. MuJoCo's upstream source is Apache-2.0; see https://github.com/google-deepmind/mujoco/blob/main/LICENSE. The simulator and its dependencies are installed separately, not vendored.

Project-authored synthetic checkpoints are distributed separately from the source in an optional release bundle. Third-party detector weights, text encoders and local recordings are not included. Original source licensing does not override the optional detector's AGPL-3.0/Enterprise terms.


## Embedded research visualization

`docs/research/flowreach-visualizations.html` embeds Plotly.js 4.1.1. Its header credits Copyright 2012–2026 Plotly, Inc.; the upstream MIT license is reproduced in `licenses/PLOTLY-MIT.txt`. Source: https://github.com/plotly/plotly.js/tree/v4.1.1. Other project plots and UI images are generated from synthetic data or reported experiments.
