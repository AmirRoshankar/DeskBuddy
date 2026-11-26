"""Feedback-gated trajectory clock with idempotent position-target refreshes."""

import time

import numpy as np

from .adaptive_control import AdaptiveCorrection
from .kinematics import JOINTS
from .state_estimation import JointKalman


def sample(plan, phase):
    index = min(
        len(plan.time_s) - 2,
        max(0, int(np.searchsorted(plan.time_s, phase, side="right") - 1)),
    )
    fraction = np.clip(
        (phase - plan.time_s[index]) / (plan.time_s[index + 1] - plan.time_s[index]),
        0,
        1,
    )
    return (
        plan.joints_rad[index] * (1 - fraction) + plan.joints_rad[index + 1] * fraction
    )


def run_feedback(
    plan,
    bias,
    limits,
    read_pose,
    write_pose,
    read_goal,
    stop_requested,
    controller,
    feedback,
    hz,
    clock=time.perf_counter,
    sleep=time.sleep,
    max_speed_deg_s=12.0,
    joint_lower=None,
    joint_upper=None,
    locked=None,
    guard_step=None,
):
    period = 1 / float(hz)
    phase = 0.0
    began = clock()
    next_tick = began
    last_good = began
    last_publish = -1.0
    last_goal_check = began
    settling_began = None
    stats = dict(
        requested_hz=float(hz),
        read_failures=0,
        write_failures=0,
        goal_read_failures=0,
        goal_mismatches=0,
        deadline_misses=0,
        iterations=0,
    )
    trace = []
    maximum = np.zeros(5)
    observed = controller["reference"].copy()
    estimator = JointKalman(observed)
    adaptive = AdaptiveCorrection(
        bias,
        controller["command"],
        np.full(5, -np.inf) if joint_lower is None else joint_lower,
        np.full(5, np.inf) if joint_upper is None else joint_upper,
        max_speed_deg_s,
        locked,
    )
    last_prediction = began
    stats.update(
        adaptive_correction=True,
        state_filter="constant_velocity_kalman",
        settling_converged=False,
    )
    last_tick = began - period
    while True:
        wait = next_tick - clock()
        if wait > 0:
            sleep(wait)
        tick = clock()
        iteration_dt = float(np.clip(tick - last_tick, period, 0.02))
        last_tick = tick
        if stop_requested():
            raise RuntimeError("Operator requested stop; position hold retained")
        if tick - last_good > 0.25:
            raise RuntimeError(
                "Motor feedback unavailable for over 250 ms; position hold retained"
            )
        stats["iterations"] += 1
        estimator.predict(max(0.0, tick - last_prediction))
        last_prediction = tick
        try:
            observed = np.asarray(read_pose(), float)
            if observed.shape != (5,) or not np.isfinite(observed).all():
                raise ValueError("Invalid joint feedback")
        except (OSError, RuntimeError, ValueError, KeyError):
            stats["read_failures"] += 1
            # Do not send a new target using blind/stale feedback or advance time.
            next_tick = max(next_tick + period, clock())
            continue
        read_ended = clock()
        last_good = read_ended
        estimated = estimator.update(observed)
        controller["estimated"] = estimated.copy()
        controller["estimation_std_deg"] = estimator.std_deg
        tracking = np.abs(observed - controller["reference"])
        maximum = np.maximum(maximum, tracking)
        error_deg = np.rad2deg(tracking)
        if (error_deg > limits).any():
            index = int(np.argmax(error_deg - limits))
            raise RuntimeError(
                f"{JOINTS[index]} tracking error {error_deg[index]:.2f}° exceeds {limits[index]:.1f}°; stopping the reach"
            )
        # Feedback every iteration; command-register verification at 10 Hz avoids
        # doubling serial traffic at 250 Hz. A broadcast write has no motor ACK.
        replay = False
        if read_ended - last_goal_check >= 0.1:
            try:
                actual_goal = read_goal()
                replay = bool(
                    np.rad2deg(np.abs(actual_goal - controller["command"])).max() > 0.25
                )
                stats["goal_mismatches"] += int(replay)
            except (OSError, RuntimeError, ValueError, KeyError):
                stats["goal_read_failures"] += 1
                replay = True
            if replay:
                if "goal_mismatch_since" not in controller:
                    controller["goal_mismatch_since"] = clock()
                if clock() - controller["goal_mismatch_since"] > 0.5:
                    raise RuntimeError(
                        "Motor goal readback did not confirm targets for over 500 ms; position hold retained"
                    )
            else:
                controller.pop("goal_mismatch_since", None)
            last_goal_check = clock()
        desired = controller["reference"].copy() if replay else sample(plan, phase)
        command, integral, error = adaptive.propose(
            desired,
            estimated,
            controller["command"],
            iteration_dt,
            freeze_integral=replay,
        )
        if not trace:
            command = controller["command"].copy()
            integral = adaptive.integral.copy()
        if guard_step is not None:
            controller["phase_s"] = phase
            guard_step(estimated, estimated + command - controller["command"])
        if np.rad2deg(np.abs(command - controller["command"])).max() > 0.4 + 1e-9:
            raise RuntimeError("Unexpected command jump; stopping the reach")
        if stop_requested():
            raise RuntimeError("Operator requested stop; position hold retained")
        # Record the attempted target even if transmission is ambiguous. Repeating
        # the same absolute position is idempotent; never accumulate relative moves.
        controller["command"] = command.copy()
        controller["reference"] = desired.copy()
        write_began = clock()
        try:
            write_pose(command)
        except (OSError, RuntimeError):
            stats["write_failures"] += 1
            if clock() - last_good > 0.25:
                raise RuntimeError(
                    "Command transmission unavailable; position hold retained"
                )
            # Freeze trajectory phase and re-evaluate/retry using the next read.
            if "write_failure_since" not in controller:
                controller["write_failure_since"] = clock()
            if clock() - controller["write_failure_since"] > 0.25:
                raise RuntimeError(
                    "Command transmission failed for over 250 ms; position hold retained"
                )
            next_tick = max(next_tick + period, clock())
            continue
        controller.pop("write_failure_since", None)
        adaptive.commit(integral)
        ended = clock()
        stats["deadline_misses"] += int(ended > next_tick + period)
        trace.append(
            dict(
                time_s=ended - began,
                commanded_rad=command.tolist(),
                reference_rad=desired.tolist(),
                observed_rad=observed.tolist(),
                estimated_rad=estimated.tolist(),
                estimation_std_deg=estimator.std_deg.tolist(),
                correction_rad=(command - desired).tolist(),
                read_ms=(read_ended - tick) * 1000,
                write_ms=(ended - write_began) * 1000,
                service_ms=(ended - tick) * 1000,
                start_lateness_ms=max(0, tick - next_tick) * 1000,
            )
        )
        if ended - began - last_publish > 0.05:
            feedback(
                observed,
                f"Reach {min(100, 5 * int(20 * phase / plan.time_s[-1]))}% | {hz:g} Hz requested | packet retries {stats['read_failures'] + stats['write_failures']} | Stop retains hold",
            )
            last_publish = ended - began
        if phase >= plan.time_s[-1] and not replay:
            if settling_began is None:
                settling_began = ended
            active = ~adaptive.locked
            settled_error = np.rad2deg(np.abs(observed - plan.joints_rad[-1]))
            converged = bool(
                (settled_error[active] < 0.20).all()
                and (np.rad2deg(np.abs(estimator.velocity[active])) < 0.5).all()
            )
            if ended - settling_began >= 0.5 and (
                converged or ended - settling_began >= 4.0
            ):
                stats["settling_converged"] = converged
                break
        elif not replay:
            # Advance by at most one nominal tick. Missed cycles never generate
            # a catch-up burst or a bigger position jump.
            # Re-time the remaining reference from current error: allow more
            # correction headroom when the physical joints fall behind.
            pace = float(np.clip(1 - max(0.0, error_deg.max() - 1.0) / 8.0, 0.15, 1.0))
            phase = min(plan.time_s[-1], phase + iteration_dt * pace)
        next_tick = max(next_tick + period, clock())
    stats["final_joint_error_deg"] = np.rad2deg(observed - plan.joints_rad[-1]).tolist()
    stats["final_estimation_std_deg"] = estimator.std_deg.tolist()
    stats["max_sent_speed_deg_s"] = (
        float(
            np.rad2deg(
                np.abs(
                    np.diff(np.array([row["commanded_rad"] for row in trace]), axis=0)
                )
            ).max()
            / period
        )
        if len(trace) > 1
        else 0.0
    )
    stats["achieved_hz"] = stats["iterations"] / (clock() - began)
    stats["successful_command_hz"] = len(trace) / (clock() - began)
    return observed, trace, maximum, stats
