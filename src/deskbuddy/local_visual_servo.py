"""Bounded image-space correction with an online local XY Jacobian."""

import numpy as np


def refine(
    observe,
    move_xy,
    stopped,
    publish=lambda report: None,
    max_steps=6,
    step_m=0.002,
    travel_m=0.010,
    tolerance_px=2.0,
    initial_jacobian=None,
):
    """Callbacks own fresh sensing and checked execution; this module has no I/O.

    observe returns error_px (observed minus desired) and a 2x2 model Jacobian
    in pixels/metre. Broyden updates use the measured achieved XY displacement.
    Only horizontal corrections are requested; image overlap never proves Z.
    """
    total = np.zeros(2)
    travelled = 0.0
    jacobian = (
        None if initial_jacobian is None else np.asarray(initial_jacobian, float).copy()
    )
    previous_error = None
    previous_move = None
    trace = []
    status = "iteration budget"
    for iteration in range(max_steps + 1):
        if stopped():
            status = "operator stopped"
            break
        sample = observe()
        if sample is None:
            status = "webcam measurement unavailable"
            break
        error = np.asarray(sample["error_px"], float)
        if error.shape != (2,) or not np.isfinite(error).all():
            status = "invalid webcam measurement"
            break
        if jacobian is None:
            jacobian = np.asarray(sample["jacobian_px_m"], float).copy()
            if jacobian.shape != (2, 2) or not np.isfinite(jacobian).all():
                status = "invalid image Jacobian"
                break
        if previous_move is not None and previous_move @ previous_move > 1e-10:
            observed_change = error - previous_error
            jacobian += np.outer(
                observed_change - jacobian @ previous_move, previous_move
            ) / (previous_move @ previous_move)
            if np.linalg.norm(error) > np.linalg.norm(previous_error) * 1.5 + 1:
                status = "image error increased; correction ended"
                break
        row = dict(
            iteration=iteration,
            error_px=float(np.linalg.norm(error)),
            travel_mm=travelled * 1000,
        )
        trace.append(row)
        publish(row)
        if np.linalg.norm(error) <= tolerance_px:
            status = "image aligned"
            break
        if iteration == max_steps or travelled >= travel_m - 1e-8:
            break
        if np.linalg.svd(jacobian, compute_uv=False)[-1] < 1:
            status = "image Jacobian poorly conditioned"
            break
        delta = -0.6 * np.linalg.solve(
            jacobian.T @ jacobian + 25 * np.eye(2), jacobian.T @ error
        )
        length = float(np.linalg.norm(delta))
        delta *= min(
            1.0,
            step_m / max(length, 1e-12),
            (travel_m - travelled) / max(length, 1e-12),
        )
        if stopped():
            status = "operator stopped"
            break
        achieved = np.asarray(move_xy(delta), float)
        if achieved.shape != (2,) or not np.isfinite(achieved).all():
            status = "invalid motion feedback"
            break
        total += delta
        travelled += float(np.linalg.norm(delta))
        if np.linalg.norm(achieved) < 0.0001:
            status = "no measured response"
            break
        previous_error, previous_move = error, achieved
    return dict(
        status=status,
        converged=status == "image aligned",
        correction_m=total.tolist(),
        travelled_m=travelled,
        trace=trace,
        depth_used=False,
    )


def measure_jacobian(observe, move_xy, stopped, probe_m=0.002):
    """Two out-and-return XY probes; use achieved encoder displacement."""
    sample = observe()
    if sample is None:
        return dict(ready=False, status="no webcam observation for probes")
    displacements, changes = [], []
    for axis in range(2):
        if stopped():
            return dict(ready=False, status="operator stopped")
        baseline = np.asarray(sample["error_px"], float)
        delta = np.zeros(2)
        delta[axis] = probe_m
        achieved = np.asarray(move_xy(delta), float)
        measured = observe() if not stopped() else None
        if stopped():
            return dict(ready=False, status="operator stopped")
        # Bound the return command too: a large encoder displacement could
        # represent sag rather than the requested probe response.
        distance = float(np.linalg.norm(achieved))
        if achieved.shape != (2,) or not np.isfinite(achieved).all():
            return dict(ready=False, status="invalid probe feedback; hold retained")
        move_xy(-achieved * min(1.0, probe_m / max(distance, 1e-12)))
        returned = observe()
        if distance > probe_m * 1.5:
            return dict(
                ready=False,
                status="probe response exceeded expected travel; no Jacobian accepted",
            )
        if measured is None or returned is None:
            return dict(
                ready=False, status="webcam lost during probes; no Jacobian accepted"
            )
        displacements.append(achieved)
        changes.append(np.asarray(measured["error_px"]) - baseline)
        sample = returned
    motion = np.column_stack(displacements)
    if (
        np.linalg.svd(motion, compute_uv=False)[-1] < 0.0004
        or np.linalg.cond(motion) > 10
    ):
        return dict(ready=False, status="probe motion too small or dependent")
    jacobian = np.column_stack(changes) @ np.linalg.inv(motion)
    if (
        not np.isfinite(jacobian).all()
        or np.linalg.svd(jacobian, compute_uv=False)[-1] < 50
        or np.linalg.cond(jacobian) > 20
    ):
        return dict(
            ready=False, status="webcam probe response too weak or ill-conditioned"
        )
    return dict(
        ready=True,
        status="measured XY image Jacobian",
        jacobian_px_m=jacobian.tolist(),
        achieved_probe_m=[p.tolist() for p in displacements],
        probe_m=probe_m,
    )
