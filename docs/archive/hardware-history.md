> Historical development record. See the current [project overview](../../docs/archive/legacy-readme.md) and [controller guide](../guides/controller.md) for supported behavior.

# Hardware checkpoint

The software environment and offline planner are ready. A read-only motor identity
diagnostic is available separately from calibration and motion:

```powershell
python -m deskbuddy.inspect_bus --port COM5
```

Substitute the confirmed adapter port. This opens only the serial bus, reads model numbers
for expected IDs 1–6 at 1 Mbaud, and closes without changing torque. It does not call the
robot's connection/configuration routine. Results are saved in `runs/motor_identity.json`.
Responses alone do not establish physical joint order or prove there are no duplicate IDs.
The diagnostic must not be confused with a motor-configuration or calibration command.

## Camera first

Identify the exact camera model using `python -m deskbuddy devices` or the printed label.
Mount the camera rigidly so the small workspace is visible. Keep the arm base rigidly clamped.
The necessary camera distance depends on the model; inspect real depth rather than guessing from RGB.

Open `python -m deskbuddy camera --config configs/local.json`.
Click broad, matte, stationary surface patches, away from boundaries.
The viewer saves each valid camera-frame XYZ to `runs/camera_clicks.jsonl` even without a transform.
Blank depth, discontinuities, and out-of-range values are rejected.

## Camera-to-base calibration

### Option A: hand-guided tool observations

Once motor calibration is saved, start
`python -m deskbuddy camera --config configs/local.json --robot-state`.
This connects the camera and the motor bus for state reads only; it does not use the
robot's configuration routine, write goals, or enable torque.

Gently close the gripper so the tips meet. Move the arm by hand to eight distinct poses,
spread across a small working volume, including different heights and several elbow/wrist
configurations. At each pose, hold it still for a second, then click the meeting point
of the jaw tips in the RGB image. That physical point must correspond to the model's tool
frame; a click elsewhere produces a different calibration.

The viewer captures depth plus the model FK position from the current calibrated joints,
and rejects moving poses, open jaws, model joint-limit violations, and near-duplicate points.
For tool captures it also requires a central image location, valid 30–100 cm depth at
the clicked pixel, agreement with nearby depth, and stable depth across recent frames.
Aim for roughly 35–65 cm working distance and use the cursor magnifier for the exact tip.
The D435's [published ideal range](https://www.realsenseai.com/products/stereo-depth-camera-d435/)
starts at 30 cm; this capture gate is deliberately conservative rather than a claim that
the camera can never return depth closer than that.
Captures are in `runs/tool_points.jsonl`; each viewer session is kept separate.
Accepted captures save annotated RGB and depth arrays in `runs/calibration/<session>/`
so outliers can be inspected. These local camera frames are excluded from Git.

After eight captures:

```powershell
python -m deskbuddy.tool_calibration --config configs/local.json
```

The first six points fit a rigid transform; the last two are excluded from fitting and
test the result. Every error must be within 12 mm by default. The command rejects inadequate
spatial spread and leaves `calibration_verified=false` even if the numerical test passes.
Review the physical tool-point location and model joint conventions, then check an additional
independent point before enabling motion. These residuals test camera/model agreement, not
absolute accuracy against external measurements. Moving the camera or arm base invalidates the fit.

The viewer publishes its current frame and read-only state in `runs/live/`. Creating
`runs/live/stop` closes it normally and releases the camera and serial bus.

### Optional model-zero diagnostic

If stable, accurately clicked observations still disagree with the URDF, the diagnostic
`python -m deskbuddy.model_calibration fit` jointly estimates camera extrinsics and small
software zero corrections for shoulder lift, elbow, and wrist flex, bounded to ±10 degrees.
It leaves base pan and wrist roll unchanged and never writes motor EEPROM or the active config.
Base-pan zero cannot be separated from the camera's global rotation using these point observations.

This is a different, more flexible model. Its training residual is not a validation result.
The candidate is saved with a hash of the exact training observations and a timestamp.
Keep the viewer and camera unchanged, capture at least two NEW poses, then run
`python -m deskbuddy.model_calibration validate`. The validator checks only those new poses
against the frozen candidate, without refitting, and retains the 12 mm check-point threshold.
Do not append points and refit repeatedly while presenting those same points as held-out tests.
A passing report tests camera/model consistency; independent measurement is still needed
to establish absolute positioning accuracy.

### Table and first preview

`table.estimate_table` uses RANSAC on calibrated depth points to identify a dominant
near-horizontal tabletop. A conservative height bound over the configured workspace
supports the existing link-origin clearance check. This is not arbitrary-object collision detection.

The first live preview locks wrist roll near the calibrated orientation and intersects
URDF bounds with the measured motor ranges after software zero corrections, including
a small margin away from physical stops. These restrictions keep the initial experiment
within the observations used for calibration. Preview mode never enables torque.

### Option B: externally measured correspondences

Use at least 6 well-spaced measured points; the solver accepts a minimum of 4.
Planar points are acceptable, but they must not be collinear or clustered together.
For each point, obtain camera XYZ from the viewer and independently measure its XYZ in the
URDF `base_link` coordinate frame. Use the model's actual origin and axes, not the corner
of the clamp or an assumed forward direction. An accurately positioned fiducial board
with a measured base-frame pose is another way to obtain these correspondences.

Copy `configs/calibration_points.template.json` to a local file and fill matching rows:
`camera_points_m[i]` and `base_points_m[i]` must refer to the same physical point, in metres.

```powershell
python -m deskbuddy calibrate runs/measured_points.json --config configs/local.json --output configs/local.json
```

This saves `T_base_camera` and reports fit residuals. It leaves `calibration_verified=false`.
Use additional points excluded from fitting to check the transform against measurements,
including height above the table. Inspect both the magnitude and direction of errors.
Only mark the calibration verified once those independent checks are acceptable.
Moving the camera or the arm base invalidates this transform.

Set the actual `table_z_m`, workspace bounds and hover clearance. The defaults describe
the offline fixture and are not measured properties of your setup.

## Arm convention and control

For an assembled but uncalibrated arm, first confirm the USB serial port and whether the
motor IDs and baud rates were configured during assembly. Do not assign IDs to a fully
connected chain of motors with duplicate factory IDs; follow the official individual
motor setup procedure if that step was skipped.

After the device and operator readiness are confirmed, `calibrate-arm.ps1 -Port COM5`
is an example invocation of the SDK calibration wrapper (substitute the
actual port). It connects the motor bus directly, backs up the existing offsets/limits,
and calls LeRobot's calibration routine without the robot connection routine's torque-enabling
configuration stage. Keyboard polling is adapted for remote stdin. Torque remains disabled.
The operator must physically position the joints and move them through
their travel when the SDK prompts. The script does not automate Enter presses or assign IDs.
Calibration is saved locally in `runs/motor-calibration/deskbuddy.json` rather than the
default user cache. A future controller must use that same calibration directory and ID.
`python -m deskbuddy.verify_calibration --port COM5` reads back the stored limits and
offsets, compares them to the saved file, and reports positions and torque state without writes.

The bundled URDF is the official **new calibration** model, where joint virtual zeros
are centered within joint ranges. Older calibration conventions can differ substantially.
Confirm the existing LeRobot robot ID/calibration file and each joint's sign and offset
before using any model angles as motor commands.

Reserved config fields specify a future adapter's mapping:

```text
q_model_deg = joint_signs * q_motor_deg + joint_offsets_deg
```

These fields currently do not activate motor control. The SDK is installed and its imports
were checked, but a controller has intentionally not been commissioned without the actual arm.
The SDK's connection routine can configure motors and change torque state, so we will inspect
calibration and the rest pose with the operator present before connecting.

The first physical test should use a clear workspace and a small, slow, reversible reach,
with the operator able to cut power. Command tracking and stopping behavior need checking
before a camera click becomes an executable target. Power removal can let the arm fall;
account for its supported rest position.

Record actual tool/target distances at several held-out positions and repeated trials.
Report calibration error, model residual and physical positioning error separately.

## Holding and Go

`python -m deskbuddy.hold_position --config configs/local.json` initializes a hold from
the measured, supported pose. Close the state viewer first so it releases the serial port.
It checks position mode and stable feedback, writes measured raw positions as goals,
reads those goals back, then enables the five arm joints. The gripper is left unchanged.
It does not execute a saved reach. Keep the support in place until hold is verified,
then withdraw the support gradually while remaining ready to support the arm again.

Reopen the live viewer with `--robot-state`. Select a target, inspect the path, and press Go.
The viewer temporarily hands the serial port to a separate motion process while the camera
and Stop button remain available. The controller leaves torque enabled after completion or stop.
It does not alter motor gains. The source pixel's depth is checked again before a Go launch.

Motor voltage and temperature are read every few seconds and logged in `runs/telemetry.jsonl`.
A controller-board input label does not establish the safe supply voltage for every servo
variant; identify the actual motor rating before changing the power supply.
