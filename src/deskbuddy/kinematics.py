"""Small URDF FK and position-only IK solver, independent of motor communication.

URDF origins compose as translation then Rz(yaw) Ry(pitch) Rx(roll).
Internal joint angles are radians. No gripper actuation is needed for reaching.
"""

import xml.etree.ElementTree as ET
from importlib.resources import files

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def axis_rotation(axis, angle):
    """Rodrigues rotation matrix; same result as Rotation.from_rotvec, ~30x cheaper."""
    axis = np.asarray(axis, float)
    norm = np.sqrt(axis @ axis)
    if norm < 1e-15:
        return np.eye(3)
    x, y, z = axis / norm
    angle = float(angle) * norm
    c, s = np.cos(angle), np.sin(angle)
    v = 1 - c
    return np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )


class Arm:
    def __init__(self, limits_deg=None, tool_offset_m=None):
        self.tool_offset = np.asarray(
            [0, 0, 0] if tool_offset_m is None else tool_offset_m, float
        )
        if (
            self.tool_offset.shape != (3,)
            or not np.isfinite(self.tool_offset).all()
            or np.linalg.norm(self.tool_offset) > 0.025
        ):
            raise ValueError("Tool offset must be a finite XYZ vector within 25 mm")
        root = ET.parse(
            files("deskbuddy").joinpath("models/so101_new_calib.urdf")
        ).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], "gripper_frame_link"
        while link != "base_link":
            j = by_child[link]
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            t = np.eye(4)
            t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            t[:3, 3] = xyz
            name = j.get("name")
            axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            limit = j.find("limit")
            bounds = (
                None
                if limit is None
                else [float(limit.get("lower")), float(limit.get("upper"))]
            )
            chain.append((name, t, axis, bounds))
            link = j.find("parent").get("link")
        self.chain = chain[::-1]
        active = [j for j in self.chain if j[3] is not None]
        if tuple(j[0] for j in active) != JOINTS:
            raise ValueError("Unexpected URDF joint order")
        self.lower, self.upper = np.array([j[3] for j in active]).T
        self.limit_source = "URDF defaults"
        if limits_deg is not None:
            bounds = np.deg2rad(np.asarray(limits_deg, float))
            if bounds.shape != (2, 5) or not np.isfinite(bounds).all():
                raise ValueError("Joint limit overrides must have shape 2x5")
            # Explicit bounds describe this commissioned arm. The generic URDF
            # remains the fallback, not a second hidden clamp on measured travel.
            self.lower, self.upper = bounds.copy()
            self.limit_source = "configured calibrated ranges"
            if (self.lower >= self.upper).any():
                raise ValueError(
                    "Configured joint lower limits must be below upper limits"
                )

    def limit_error(self, pose, label="Initial joints"):
        degrees = np.rad2deg(pose)
        lower, upper = np.rad2deg([self.lower, self.upper])
        details = [
            f"{name} {value:.2f}° (allowed {lo:.2f}–{hi:.2f}°)"
            for name, value, lo, hi in zip(JOINTS, degrees, lower, upper)
            if value < lo or value > hi
        ]
        return (
            f"{label} outside model limits: "
            + "; ".join(details)
            + f" [{self.limit_source}]"
        )

    def frames(self, q):
        q = np.asarray(q, float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("Expected five finite joint angles in radians")
        t, points, i = np.eye(4), [np.zeros(3)], 0
        for _, origin, axis, bounds in self.chain:
            t = t @ origin
            if bounds is not None:
                motion = np.eye(4)
                motion[:3, :3] = axis_rotation(axis, q[i])
                t = t @ motion
                i += 1
            points.append(t[:3, 3].copy())
        if np.any(self.tool_offset):
            t[:3, 3] += t[:3, :3] @ self.tool_offset
            points.append(t[:3, 3].copy())
        return np.array(points), t

    def fk(self, q):
        return self.frames(q)[1][:3, 3]

    def ik(self, target, seed, tolerance=0.002, locked_joints=(), initial_guess=None):
        target, seed = np.asarray(target, float), np.asarray(seed, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target must be finite XYZ")
        self.frames(seed)
        if (seed < self.lower).any() or (seed > self.upper).any():
            raise ValueError(self.limit_error(seed, "IK seed"))
        unknown = set(locked_joints) - set(JOINTS)
        if unknown:
            raise ValueError(f"Unknown locked joints: {unknown}")
        free = np.array(
            [i for i, name in enumerate(JOINTS) if name not in locked_joints]
        )
        if not len(free):
            raise ValueError("At least one joint must be free for IK")
        seed = seed.copy()
        guess = seed if initial_guess is None else np.asarray(initial_guess, float)
        self.frames(guess)
        initial = np.clip(guess[free], self.lower[free] + 1e-8, self.upper[free] - 1e-8)

        def expand(values):
            full = seed.copy()
            full[free] = values
            return full

        # Small posture penalty selects a nearby solution for this redundant position task.
        def residual(values):
            q = expand(values)
            return np.r_[self.fk(q) - target, 0.0001 * (q - seed)]

        result = least_squares(
            residual,
            initial,
            bounds=(self.lower[free], self.upper[free]),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=160,
        )
        solution = expand(result.x)
        error = np.linalg.norm(self.fk(solution) - target)
        if not result.success or error > tolerance:
            # A tip behind the base can be reached by folding over the shoulder,
            # without rotating pan toward the target azimuth. Try other posture
            # branches only when the fast local solve fails.
            candidates = np.random.default_rng(101).uniform(
                self.lower, self.upper, (24 if target[0] < 0 else 0, 5)
            )
            for posture in candidates:
                trial = least_squares(
                    residual,
                    np.clip(
                        posture[free], self.lower[free] + 1e-8, self.upper[free] - 1e-8
                    ),
                    bounds=(self.lower[free], self.upper[free]),
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=120,
                )
                alternative = expand(trial.x)
                gap = np.linalg.norm(self.fk(alternative) - target)
                if gap < error:
                    solution, error = alternative, gap
                if trial.success and gap <= tolerance:
                    return alternative
            raise ValueError(
                f"IK failed across posture branches: endpoint error {error * 1000:.1f} mm"
            )
        return solution
