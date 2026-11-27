"""Select calibration joint poses only after validating whole connecting routes."""

import numpy as np


def choose_poses(arm, start, config, plan_route, count=15):
    from .trajectory.joint_planner import JointSpace

    cfg = dict(config, locked_joints=[], allow_object_collisions=False)
    space = JointSpace(arm, cfg, [], start)
    candidates = []
    for degrees in (8, 6, 4, 2):
        for joint in range(5):
            for sign in (-1, 1):
                offset = np.zeros(5)
                offset[joint] = np.deg2rad(sign * degrees)
                candidates.append(start + offset)
    rng = np.random.default_rng(101)
    candidates.extend(start + np.deg2rad(rng.uniform(-6, 6, (45, 5))))
    accepted = [np.asarray(start).copy()]
    rejected = []
    for goal in candidates:
        reason = space.reason(goal)
        if reason:
            rejected.append(reason)
            continue
        if min(np.linalg.norm(goal - p) for p in accepted) < np.deg2rad(1):
            continue
        try:
            plan_route(accepted[-1], goal)
        except ValueError as exc:
            rejected.append(str(exc))
            continue
        accepted.append(goal.copy())
        if len(accepted) == count:
            break
    if len(accepted) < 12:
        from collections import Counter

        summary = "; ".join(
            f"{n} rejected: {reason}" for reason, n in Counter(rejected).most_common(2)
        )
        raise ValueError(
            f"Only {len(accepted)} route-checked poses (need 12). {summary}"
        )
    return accepted, dict(
        accepted=len(accepted),
        rejected=len(rejected),
        max_offset_deg=float(np.rad2deg(np.max(np.abs(np.asarray(accepted) - start)))),
    )
