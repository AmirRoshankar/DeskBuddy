# Controller

The 3D view, camera overlays and planner use one robot-base world frame. Their geometry is estimated; inspect calibration and encoder agreement before relying on it.

## Select and preview

Click in 3D, color, depth or webcam view to select a world target. Ordinary wheel changes continuous bounded hover height. Ctrl+left drag rotates, Ctrl+right drag pans, Ctrl+wheel zooms; Ctrl+double-left resets the view. Ctrl navigation does not choose targets.

The sidebar selects the execution planner and one/four candidates. Direct (white), MLP (gold), iMF (purple) and whole-arm RRT (teal) previews appear together. Dashed paths failed checks; stale paths are marked. Go executes the selected checked route; it does not silently substitute another planner. Completed/stopped paths clear.

## Motion controls

- **Enable** records the supported current pose and restores holding. Complete the support/hands-clear check before motion.
- **Disable releases torque on all six servos**, permitting manual repositioning. Support the arm first.
- **Stop** ends a motion while retaining position hold.
- **Go** starts the previewed reach. Startup, target selection, camera reconnection and bounds editing do not start motion.
- **Home** plans a nominal closed-gripper rest pose with its own workspace/object policy; actual joint/self/feedback checks remain.

Tracking error means commanded/reference joint angle minus encoder feedback, not the size of a planned step. Loop rate, joint speed, command-step guard and tracking threshold are separate quantities. The requested 250 Hz is measured at runtime and not guaranteed.

## Calibration and bounds

Recalibrate guides hand-collected arm/camera correspondences, automatically assessing eight captures and independent checks. Calibrate views uses draggable ordered A4 corners for the second camera. Save-with-warnings is available; elevated geometry can remain unverified. Moving a camera requires updated alignment.

Bounds defines robot-base XYZ limits and an optional paper inset. A visible 3D volume click is intersected with that volume, rather than rejected by projecting it to an unrelated plane. A valid tip position still requires a reachable joint pose and checked arm path.

## Cans

Calibrate a discard zone using four ordered, noncrossing table corners. The body baseline and iMF rim action automatically detect a single upright can and preflight the complete sequence. The RGB cylinder prior assumes 66 mm diameter and 122.22 mm height; it does not measure these dimensions from depth.

After lift, fresh raw RGB appearance is compared against table-stationary and jaw-attached hypotheses. Confirmed miss retries a distinct approach, up to five total attempts. Ambiguity stops into hold without opening a possibly held can. Confirmed pickup permits transfer/release; landing is not independently verified. Every retry is replanned from measured joints. The physics-trained checkpoint is a separate research result, not automatically the live default.

These grasp/retry behaviors have offline tests but no validated physical rim-grasp success. Read [the experiment and limitations](can-physics-and-retries.md).
