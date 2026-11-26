# Physical reaching error: investigation and choices

The operator reports that the physical gripper misses the requested location.
The camera's magenta tip marker is an encoder/FK projection, not independent
visual tracking. Making the displayed model error small cannot by itself prove
that the physical tip reached the target.

## What the code and latest trials show

- `control_plan.py` measures `bias = measured - held_command` once at Go.
- `feedback_loop.py` uses `command = desired - bias` throughout the reach. It
  checks current feedback, stops on excessive tracking error, and retries lost
  communication, but does not adapt the correction to changing joint error.
- In one recent trial, elbow bias began at +4.84 degrees. Final observed elbow
  error relative to the physical reference was -4.65 degrees. Thus the actual
  encoder-to-command offset at the end was only approximately +0.19 degrees.
  Applying the initial correction throughout overcompensated at the endpoint.
- In another trial, initial elbow bias was +0.44 degrees and final reference
  error was +4.39 degrees: the required correction changed in the other direction.
- The five latest completed trials reported encoder/model endpoint gaps of
  9.0, 22.0, 14.1, 13.6 and 13.2 mm. These are not external measurements.
- Those trials achieved approximately 249.9 Hz with no recorded read failures,
  write failures, goal-read failures or goal mismatches. Occasional deadline
  misses occurred, but packet loss is not supported as the main explanation.
- Read-only servo inspection confirmed P=16, I=0 and D=32. No gains were changed.
- A4 pose fitting previously disagreed with corner depth by 7–18 mm. The
  calibration's fresh tip checks were 5.9 and 3.4 mm against model-derived
  positions. Neither establishes uniform external accuracy throughout the bed.

## Options for the operator to choose

| Option | Change | What it addresses | Limitation | Rough effort |
| --- | --- | --- | --- | --- |
| 1. Adaptive joint correction | Replace the frozen bias with a bounded, filtered joint-error feedback correction, including carefully limited integral action and anti-windup. | Demonstrated pose-dependent over/undercompensation and persistent joint error. | Encoders cannot see an incorrect camera transform, tip geometry, flex or backlash outside their measurement. | 1–2 hours |
| 2. Actual tip tracking and visual servoing | Attach a known-size ArUco marker to the gripper, calibrate its offset to the tip, observe its RGB pose, and make small checked corrections relative to the target. | Physical misses that encoder/FK predictions cannot independently observe. | Requires marker visibility and measured camera/marker calibration; precision must be validated. A missing marker may use prediction temporarily, but cannot prove final visual success. | 2–4 hours |
| 3. Independent geometry calibration | Use a measured fiducial board on the bed and an observed tip reference to check camera pose, distortion handling and tool offset. | Consistent spatial misalignment of targets and tip projections. | Does not alone correct load-dependent servo tracking error. | 0.5–1.5 hours |
| 4. Onboard servo tuning | Perform controlled single-joint response tests and tune existing position-loop gains from measured behavior. | Holding deflection, stiffness and lag inside the servos. | Incorrect gains can cause oscillation; this does not fix camera/model geometry. | 1–2 hours |

Effort estimates are implementation estimates, not measured outcomes. These
methods can be combined. The immediate code fix is option 1; option 2 adds the
independent physical observation needed to assess and correct actual landing.
The operator will select the correction strategy before implementation.

## Primary references and how they apply

- [Modern Robotics: feedforward plus PI feedback](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-3-motion-control-with-velocity-inputs-part-2-of-3/)
  explains using measured error and integral action to remove steady error.
  Its example uses velocity inputs; our nested position-servo application needs
  separate gain selection, rate limits and validation, rather than copied gains.
- [Modern Robotics: joints under gravity](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-4-motion-control-with-torque-or-force-inputs-part-2-of-3/)
  explains how gravity produces holding error and why integral tuning has
  stability constraints. A changing load/pose is a plausible mechanism here;
  the logs demonstrate changing bias, not a complete identification of its cause.
- [Chaumette and Hutchinson: Visual Servo Control](https://faculty.cc.gatech.edu/~seth/ResPages/pdfs/ChaHut06.pdf)
  describes closing the motion loop using measured image or estimated 3D features,
  including a fixed camera observing robot motion.
- [OpenCV: ArUco detection and pose estimation](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html)
  provides marker-based RGB pose estimation using known marker size, camera
  intrinsics and distortion coefficients, without requiring live stereo depth.
- [LeRobot follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
  exposes position P/I/D gains. [FEETECH's STS register documentation](https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/)
  documents those registers, integral limits and dead zones. Their availability
  does not establish suitable gains for this particular arm and load.

## Speed change applied separately

Half-step subdivision now preserves the original trajectory duration. The
configured hardware trajectory ceiling is approximately 3 degrees/s rather than
the previous effective 1.5 degrees/s. The requested control frequency remains
250 Hz. All 87 tests and lint checks passed. No movement was started by this
investigation, and no PID, torque, voltage or correction-policy changes were made.
