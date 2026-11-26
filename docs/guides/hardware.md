# Hardware and calibration

Developed with an SO-101 follower (five arm joints plus gripper), STS3215 servos, a RealSense D435 and a Logitech Brio RGB webcam. ROS 2 is not required; LeRobot supplies the serial transport.

Commission motor IDs and encoder ranges with the operator present before motion. Calibration changes motor settings/torque and requires hand positioning; `scripts/calibrate-arm.ps1` is an explicit hardware operation, not an installation check. Support the arm before releasing hold. Use the rating on the actual servos and identify the supply; the controller-board marking is not a servo rating.

Keep `configs/local.json` and `runs/motor-calibration/` local. Calibration layers are separate:

1. Motor IDs, directions, encoder ranges and model joint mapping.
2. URDF joint-zero/tool-point correction.
3. Camera-to-base correspondences with independent check points.
4. Metric A4 paper/table plane and the second camera's ordered corner alignment.

Moving a camera invalidates its alignment. A saved planar homography may be useful despite camera-pose warnings, but raised 3D geometry remains uncertain. The optional level-table assumption must match the actual scene. Do not change joint ranges solely because a camera target looks reachable.

Startup/passive reads do not enable torque. Disable releases the arm; Stop retains hold. Do not run separate serial tools while a worker owns `runs/control/pause_robot`.

Useful read-only entry points:

```bash
python -m deskbuddy devices
python -m deskbuddy --help
```

`doctor --require-cuda` verifies environment/GPU computation; it is not commissioning or calibration. Older commissioning transcripts are retained under `docs/archive/` and are not current operating instructions.
