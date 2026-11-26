> Historical development record. See the current [project overview](../../docs/archive/legacy-readme.md) and [controller guide](../guides/controller.md) for supported behavior.

# FlowReach experiment after the baseline

Goal: learn a conditional distribution over reaching trajectories, then compare it with
the deterministic baseline under identical target and validation conditions.
This experiment is planned; there is no trained flow model in this checkpoint.

## Small first formulation

- Condition on the 5D start pose and 3D goal position, with fixed gripper state.
- Generate residuals for a short set of interior joint-space waypoints relative to the baseline.
- Fix the start/end configurations explicitly. The baseline still solves target IK.
- Normalize angles and use a small MLP conditioned on interpolation time and the task condition.
- Integrate the learned vector field with a small fixed-step ODE solver.
- Resample the resulting curve at the controller rate and pass the same validator.

For normalized data trajectory `x1`, Gaussian noise `x0`, and `t ∈ [0,1]`:

```text
xt = (1-t) x0 + t x1
loss = mean_square(v_theta(xt, t, condition) - (x1-x0))
```

At inference integrate `dx/dt = v_theta(x,t,condition)` from noise at time zero.
Flow integration time is a generative-model variable, not physical execution time.

## Dataset and comparison

Start with synthetic teacher paths generated from reachable model poses, including a small
range of validated posture variations. Split by targets, not adjacent frames from the same
trajectory. A teacher consisting only of one quintic interpolation per task mostly teaches
the network to imitate a function we already have; document that limitation.

Use the same held-out target list and multiple sample seeds. Report invalid samples in the
denominator rather than silently filtering them out. Measure valid-plan rate, endpoint error,
joint speed and acceleration, path length, and latency. Compare to both the baseline and
a simple interpolation or regression ablation when useful.

After offline checks, evaluate only validated trajectories on the commissioned arm, using
the same workspace and controller limits. Physical data and learned success rates should
only enter the README after they have actually been measured.
