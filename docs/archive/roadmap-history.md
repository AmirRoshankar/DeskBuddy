> Historical development record. See the current [project overview](../../docs/archive/legacy-readme.md) and [controller guide](../guides/controller.md) for supported behavior.

# DeskBuddy: handoff reviewed against the working system

The supplied research handoff is a useful direction, rather than a description of
the current implementation. Its external literature summaries remain unverified
here; do not copy their dates, novelty claims or quantitative results into a paper.

Progress: the offline 2D teacher pipeline and first four-model trajectory benchmark
are now complete; see [results and limitations](../research/trajectory-learning.md). Obstacle
perception, joint lifting, physical trials and dynamic replanning remain future work.

## What already exists

- SO-101 joint-angle commands, calibrated encoder feedback, URDF FK and bounded IK.
- A calibrated camera-to-base transform and fitted tabletop plane. Clicks use
  ray/plane intersection on the test bed, with cached depth elsewhere.
- A requested 250 Hz joint feedback controller with filtering, tracking checks,
  retries, Stop and a persistent Disable control. The UI launches one checked reach
  at a time; replacing a running trajectory is not implemented.
- An MLP and actual iMF objective for **joint-goal proposals**, followed by numerical
  IK refinement and quintic interpolation. These models do not generate complete
  obstacle-aware trajectories. Three small physical learned-model reaches have run.
- Model-projected tip overlays, not independent visual tracking of the physical tip.
  Link-origin/table checks are screening, not full-arm collision guarantees.

## Recommended next experiment

Keep the working controller. Build a separate, offline 2D trajectory benchmark:

1. Implement circle/rectangle obstacle inflation and exact segment collision tests,
   then RRT-Connect, collision-checked shortcutting and 32-point resampling. Recheck
   the resampled segments; interpolation can otherwise cut through corners.
2. Generate shared scenarios with repeated seeds per condition so both routes
   around an obstacle are represented. Split by scene/configuration family before
   training, keeping all routes for one condition in the same split. Retain failed
   teacher cases in evaluation accounting rather than hiding them.
3. Train a conditional MLP trajectory baseline, conditional FM, MeanFlow and iMF
   with matched data, conditioning and approximately matched capacity. Enforce
   endpoints consistently. Keep the existing joint-goal models as a separate task.
4. Compare quality versus NFE/latency: raw validity, validator rejection, goal
   error, clearance, length and smoothness; include fallback and validation time.
   Motor energy remains an explicitly estimated metric until current is measured.

Then add one manually selected static obstacle to the camera pipeline. Measure its
localization error against physical dimensions before choosing inflation margins.
Lift Cartesian paths into continuous, branch-consistent joint paths using existing
IK, and validate the interpolated arm motion, speed and acceleration. Cartesian
tool-path validity alone does not establish robot-link clearance.

Only after static validation and supervised execution should stop-and-replan be
added: stop first, estimate the current state, update obstacles, generate and
validate a replacement, then explicitly resume. Do not send independent blocking
point-to-point jobs for every waypoint: that would add stops and lose continuity.

## Research question worth testing

Does iMF reduce **end-to-end validated replanning latency** at comparable success
and clearance, rather than merely reducing network evaluations? Investigate
uncertainty conditioning and continuity as later ablations against conservative
geometric inflation and fresh-noise generation. Neither novelty nor superiority
is established by this handoff or the current small hardware trial.

Keep generated artifacts under the existing 10 GB budget. Simulation and training
must remain independent of motor commands. This roadmap is not an implementation
or an authorization to execute obstacle experiments on hardware.
