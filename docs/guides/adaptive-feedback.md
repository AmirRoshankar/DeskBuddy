# Adaptive feedback and estimation

The command speed ceiling is 12 degrees/s, independently of the requested 250 Hz
feedback rate. This is an experimental outer controller around the existing servo
position controllers; no motor firmware gains or voltage settings are changed.

## Joint loop

Each valid encoder reading updates five independent constant-velocity Kalman filters.
Assumed encoder standard deviation is 0.05 degrees and acceleration process noise
is 40 degrees/s². These are tuning assumptions, not measured sensor specifications.
The predicted angle and velocity remain uncertain during missing reads; the loop
sends no new command without valid encoder feedback.

The initial integral correction cancels measured holding bias, preserving the existing
first hold command. Thereafter command = reference + integral + 0.2 × position error,
with integral gain 1/s. Integral correction is bounded by at least ±12 degrees,
with conditional anti-windup at command and rate limits. Failed writes do not commit
integral updates. Wrist roll remains locked. The complete command, including
correction, is rate-limited to 12 degrees/s and 0.4 degrees per cycle.

Every iteration checks raw tracking error against the unchanged 10-degree limit.
Reference time advances more slowly above 1 degree of tracking lag, down to 15% of
nominal pace. The desired path and endpoint do not drift toward the measured pose.
Settling requires all unlocked joints below 0.20 degrees error and 0.5 degrees/s
estimated velocity after at least 0.5 s; after 4 s it reports whether this converged.
Execution completion and model endpoint success are reported separately.

Corrected steps check command joint limits and predicted link-origin table clearance.
A planned initial clearance shortfall up to 5 mm permits only recovery paths whose
shortfall never increases and ends at zero. These are sampled model checks, not
mesh collision detection or a measured clearance guarantee.

## Depth map

Each camera pixel has a scalar constant-position Kalman depth estimate. Process
variance grows at (1 mm)²/s. Measurement standard deviation is assumed to be
2 mm + 4 mm × depth², with depth in metres. Invalid observations preserve the
estimate and grow covariance. Innovations above max(15 mm, 4 standard deviations)
are withheld unless three consistent readings confirm a changed surface.

The fitted A4 plane supplies paper targets even through depth holes. The map does
not repair an incorrect camera extrinsic transform or invent unseen geometry.

## Verification and physical limits

Tests use simulated servo lag, changing holding deflection, noisy encoders, packet
loss and depth noise/spikes. Real-arm stability and physical endpoint accuracy with
this controller still require operator-initiated trials. A Kalman filter improves a
noisy estimate; it does not fix systematic calibration errors.

The orange fixed-jaw marker and blue paper cross are operator-provided calibration
cues. Their detection and fusion are not implemented yet. The orange marker center
must be calibrated relative to the jaw tip before it can measure tool error. The
current magenta tip overlay is still encoder/FK-derived.
