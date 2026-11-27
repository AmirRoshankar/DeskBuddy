"""Reach envelope of the calibrated tool tip from URDF link geometry and joint ranges.

Kinematic reach only (link lengths, measured joint ranges, calibrated tool
offset, a fixed wrist roll). Table, object and self-collision clearance are
separate checks: a reachable point can still be blocked by them.

The envelope is sampled in the shoulder-pan frame, which removes one joint:
the pan joint only rotates the (radius, height) set about its vertical axis.
Lift, elbow and wrist flex are sampled on a uniform grid. A Lipschitz bound on
how far the tip moves between neighbouring samples turns the samples into
certified statements:

* ``unreachable``: farther than that bound from every sample, so no joint
  configuration inside the ranges reaches it. Provably outside.
* ``reachable``: an IK solution inside the joint ranges was found, seeded from
  the nearest samples, within the IK tolerance. Proven by that solution.
* ``boundary``: the thin band in between (a few millimetres at the edge).
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np

from .kinematics import JOINTS

CELL_M = 0.002
STEP_DEG = 1.0
SECTORS = 16
SECTOR = 2 * np.pi / SECTORS


def _batched_tip(arm, q):
    """Vectorized FK of the calibrated tip for joint rows ``q`` (N, 5)."""
    count = len(q)
    t = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    index = 0
    for _, origin, axis, bounds in arm.chain:
        t = t @ origin
        if bounds is not None:
            r = axis_rotation_batch(axis, q[:, index])
            t[:, :3, :3] = t[:, :3, :3] @ r
            index += 1
    return t[:, :3, 3] + t[:, :3, :3] @ arm.tool_offset


def _pan_axis(arm):
    name, origin, axis, _ = arm.chain[0]
    if name != JOINTS[0]:
        raise ValueError("Expected shoulder pan as the first joint")
    direction = origin[:3, :3] @ (axis / np.linalg.norm(axis))
    if abs(direction[2]) < 0.999:
        raise ValueError("Reach map assumes a vertical shoulder-pan axis")
    return origin[:3, 3].copy(), float(np.sign(direction[2]))


def _sampled_levers(arm, q):
    """Largest distance from the lift, elbow and wrist-flex axes to the tip."""
    count = len(q)
    t = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    index, frames = 0, []
    for _, origin, axis, bounds in arm.chain:
        t = t @ origin
        if bounds is not None:
            frames.append(
                (t[:, :3, 3].copy(), t[:, :3, :3] @ (axis / np.linalg.norm(axis)))
            )
            q_rot = q[:, index]
            r = axis_rotation_batch(axis, q_rot)
            t[:, :3, :3] = t[:, :3, :3] @ r
            index += 1
    tip = t[:, :3, 3] + t[:, :3, :3] @ arm.tool_offset
    levers = []
    for joint in (1, 2, 3):
        point, direction = frames[joint]
        offset = tip - point
        along = (offset * direction).sum(1, keepdims=True) * direction
        levers.append(np.linalg.norm(offset - along, axis=1).max())
    return np.asarray(levers)


def axis_rotation_batch(axis, angles):
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angles), np.sin(angles)
    v = 1 - c
    r = np.empty((len(angles), 3, 3))
    r[:, 0, 0] = c + x * x * v
    r[:, 0, 1] = x * y * v - z * s
    r[:, 0, 2] = x * z * v + y * s
    r[:, 1, 0] = y * x * v + z * s
    r[:, 1, 1] = c + y * y * v
    r[:, 1, 2] = y * z * v - x * s
    r[:, 2, 0] = z * x * v - y * s
    r[:, 2, 1] = z * y * v + x * s
    r[:, 2, 2] = c + z * z * v
    return r


class ReachMap:
    def __init__(self, arm, roll_rad, step_deg=STEP_DEG, cell_m=CELL_M):
        self.arm, self.roll = arm, float(roll_rad)
        self.cell, self.step = cell_m, np.deg2rad(step_deg)
        self.center, self.sign = _pan_axis(arm)
        lo, hi = arm.lower.copy(), arm.upper.copy()
        self.pan_low, self.pan_high = lo[0], hi[0]
        grids = [
            np.linspace(
                lo[j], hi[j], max(2, int(np.ceil((hi[j] - lo[j]) / self.step)) + 1)
            )
            for j in (1, 2, 3)
        ]
        spacing = np.array([g[1] - g[0] for g in grids])
        lift, elbow, flex = np.meshgrid(*grids, indexing="ij")
        q = np.zeros((lift.size, 5))
        q[:, 1], q[:, 2], q[:, 3], q[:, 4] = (
            lift.ravel(),
            elbow.ravel(),
            flex.ravel(),
            self.roll,
        )
        tips = np.concatenate(
            [_batched_tip(arm, q[i : i + 200_000]) for i in range(0, len(q), 200_000)]
        )
        # Any configuration is within half a grid step of a sample per joint;
        # the tip then moves at most lever_j * step_j / 2 per joint, where
        # lever_j is the largest joint-axis-to-tip distance (sampled, +10%).
        levers = _sampled_levers(arm, q[:: max(1, len(q) // 50_000)])
        self.sample_error = float((1.1 * levers) @ (spacing / 2))
        rel = tips - self.center
        radius = np.hypot(rel[:, 0], rel[:, 1])
        alpha = np.arctan2(rel[:, 1], rel[:, 0]) * self.sign
        height = tips[:, 2]
        self.r0, self.z0 = 0.0, float(height.min()) - 4 * cell_m
        shape = (
            int(np.ceil(radius.max() / cell_m)) + 8,
            int(np.ceil((height.max() - self.z0) / cell_m)) + 8,
        )
        ir = np.floor(radius / cell_m).astype(int)
        iz = np.floor((height - self.z0) / cell_m).astype(int)
        # The tip's angular offset from the pan direction is not one interval:
        # leaning back over the base gives offsets near 180 degrees. Keep a
        # per-cell set of 22.5 degree sectors instead of a min/max range.
        sector = np.floor((alpha + np.pi) / SECTOR).astype(int) % SECTORS
        self.shape = shape
        flat = (ir * shape[1] + iz) * SECTORS + sector
        size = shape[0] * shape[1] * SECTORS
        occupied = np.zeros(size, bool)
        occupied[flat] = True
        seed = np.full((size, 4), np.nan, np.float32)
        seed[flat, :3] = q[:, 1:4]
        seed[flat, 3] = alpha
        self.occupied = occupied.reshape(shape + (SECTORS,))
        self.seed = seed.reshape(shape + (SECTORS, 4))
        reach = int(np.ceil(self.sample_error / cell_m)) + 1
        self.outer = np.stack(
            [_dilate(self.occupied[..., k], reach) for k in range(SECTORS)], -1
        )
        self.dilation_m = reach * cell_m
        self.samples = len(q)

    # -- geometry ---------------------------------------------------------
    def _cyl(self, point):
        p = np.asarray(point, float) - self.center
        return (
            float(np.hypot(p[0], p[1])),
            float(point[2]),
            float(np.arctan2(p[1], p[0]) * self.sign),
        )

    def _cell(self, radius, height):
        return (
            int(np.floor(radius / self.cell)),
            int(np.floor((height - self.z0) / self.cell)),
        )

    def _inside(self, index):
        return 0 <= index[0] < self.shape[0] and 0 <= index[1] < self.shape[1]

    _ARRAYS = ("occupied", "outer", "seed")

    def save(self, path):
        np.savez_compressed(
            path,
            **{name: getattr(self, name) for name in self._ARRAYS},
            meta=np.array(
                json.dumps(
                    dict(
                        roll=self.roll,
                        cell=self.cell,
                        step=self.step,
                        center=self.center.tolist(),
                        sign=self.sign,
                        pan=[self.pan_low, self.pan_high],
                        z0=self.z0,
                        shape=list(self.shape),
                        sample_error=self.sample_error,
                        dilation_m=self.dilation_m,
                        samples=self.samples,
                    )
                )
            ),
        )

    @classmethod
    def load(cls, arm, path):
        self = cls.__new__(cls)
        with np.load(path) as data:
            for name in cls._ARRAYS:
                setattr(self, name, data[name])
            meta = json.loads(str(data["meta"]))
        self.arm, self.roll, self.cell, self.step = (
            arm,
            meta["roll"],
            meta["cell"],
            meta["step"],
        )
        self.center, self.sign = np.asarray(meta["center"]), meta["sign"]
        self.pan_low, self.pan_high = meta["pan"]
        self.z0, self.shape = meta["z0"], tuple(meta["shape"])
        self.sample_error, self.dilation_m = meta["sample_error"], meta["dilation_m"]
        self.samples, self.r0 = meta["samples"], 0.0
        return self

    def _sector_ok(self, phi, k, slack):
        """Can pan place a sector-k offset at azimuth phi (with angular slack)?"""
        low = -np.pi + k * SECTOR - slack
        width = SECTOR + 2 * slack
        # Need alpha in [low, low+width] with phi - alpha in [pan_low, pan_high]:
        # alpha in [phi - pan_high, phi - pan_low]. Intersect two arcs.
        return _arcs_meet(low, width, phi - self.pan_high, self.pan_high - self.pan_low)

    def outside(self, point):
        """True only when the point is provably beyond reach."""
        radius, height, phi = self._cyl(point)
        index = self._cell(radius, height)
        if not self._inside(index):
            return True
        sectors = np.flatnonzero(self.outer[index])
        if not len(sectors):
            return True
        slack = min(np.pi, self.dilation_m / max(radius, 1e-3))
        return not any(self._sector_ok(phi, k, slack) for k in sectors)

    def certify(self, point, tolerance=0.002):
        """Joint solution inside the ranges reaching ``point``, else None."""
        if self.outside(point):
            return None
        radius, height, phi = self._cyl(point)
        index = self._cell(radius, height)
        arm = self.arm
        seeds = []
        for di in (0, -1, 1, -2, 2, -3, 3):
            for dj in (0, -1, 1, -2, 2, -3, 3):
                cell = (index[0] + di, index[1] + dj)
                if not self._inside(cell):
                    continue
                for k in np.flatnonzero(self.occupied[cell]):
                    joints = self.seed[cell][k]
                    pan = _wrap(phi - float(joints[3])) * self.sign
                    if arm.lower[0] - 0.05 <= pan <= arm.upper[0] + 0.05:
                        seeds.append(
                            (abs(di) + abs(dj), np.r_[pan, joints[:3], self.roll])
                        )
        seeds.sort(key=lambda item: item[0])
        guesses = [guess for _, guess in seeds[:8]]
        # Near the edge of reach nearby samples can all sit in the wrong IK
        # branch; deterministic spread seeds settle most of those points.
        spread = np.random.default_rng(7).uniform(arm.lower, arm.upper, (48, 5))
        spread[:, 4] = self.roll
        for guess in [*guesses, *spread]:
            guess = np.clip(guess, arm.lower + 1e-7, arm.upper - 1e-7)
            try:
                return arm.ik(point, guess, tolerance, locked_joints=["wrist_roll"])
            except ValueError:
                continue
        return None

    def classify(self, point, tolerance=0.002):
        if self.outside(point):
            return "unreachable"
        return "reachable" if self.certify(point, tolerance) is not None else "boundary"

    def distance_outside(self, point):
        """Lower bound on how far beyond reach in the radius/height plane."""
        if not self.outside(point):
            return 0.0
        radius, height, _ = self._cyl(point)
        rs, zs = np.nonzero(self.occupied.any(-1))
        dr = (rs + 0.5) * self.cell - radius
        dz = (zs + 0.5) * self.cell + self.z0 - height
        planar = np.hypot(dr, dz) - self.cell - self.sample_error
        return float(max(0.0, planar.min()))

    def bounds(self, floor_z=None):
        """Axis-aligned box containing every reachable point (certified superset)."""
        mask = self.outer.any(-1)
        rs, zs = np.nonzero(mask)
        r = (rs.max() + 1) * self.cell
        z = np.array([zs.min(), zs.max() + 1]) * self.cell + self.z0
        if floor_z is not None:
            z[0] = max(z[0], floor_z)
        sectors = np.flatnonzero(self.outer.any((0, 1)))
        phis = np.linspace(-np.pi, np.pi, 1441)
        ok = [any(self._sector_ok(p, k, 0.02) for k in sectors) for p in phis]
        phis = phis[ok] * self.sign
        xs = np.r_[0.0, r * np.cos(phis)]
        ys = np.r_[0.0, r * np.sin(phis)]
        return (
            [
                float(self.center[0] + xs.min()),
                float(self.center[1] + ys.min()),
                float(z[0]),
            ],
            [
                float(self.center[0] + xs.max()),
                float(self.center[1] + ys.max()),
                float(z[1]),
            ],
        )

    def summary(self):
        rs, zs = np.nonzero(self.occupied.any(-1))
        return dict(
            samples=self.samples,
            cell_mm=self.cell * 1000,
            sample_error_mm=round(self.sample_error * 1000, 2),
            max_radius_m=float((rs.max() + 1) * self.cell),
            height_range_m=[
                float(zs.min() * self.cell + self.z0),
                float((zs.max() + 1) * self.cell + self.z0),
            ],
            pan_range_deg=np.rad2deg([self.pan_low, self.pan_high]).tolist(),
            wrist_roll_deg=float(np.rad2deg(self.roll)),
        )


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _arcs_meet(a_start, a_width, b_start, b_width):
    if a_width >= 2 * np.pi or b_width >= 2 * np.pi:
        return True
    offset = (b_start - a_start) % (2 * np.pi)
    return offset <= a_width or offset >= 2 * np.pi - b_width


def _dilate(mask, radius):
    out = mask.copy()
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di * di + dj * dj > radius * radius:
                continue
            out |= _shift(mask, di, dj, False)
    return out


def _shift(array, di, dj, fill):
    out = np.full_like(array, fill)
    h, w = array.shape
    src_i = slice(max(0, -di), min(h, h - di))
    dst_i = slice(max(0, di), min(h, h + di))
    src_j = slice(max(0, -dj), min(w, w - dj))
    dst_j = slice(max(0, dj), min(w, w + dj))
    out[dst_i, dst_j] = array[src_i, src_j]
    return out


_CACHE = {}
_BUILDING = {}
_LOCK = threading.Lock()
CACHE_DIR = Path("runs/reach-cache")


def _key(arm, roll_deg):
    return hashlib.sha1(
        json.dumps(
            dict(
                version=2,
                lower=np.round(arm.lower, 9).tolist(),
                upper=np.round(arm.upper, 9).tolist(),
                tool=np.round(arm.tool_offset, 9).tolist(),
                chain=[np.round(o, 9).tolist() for _, o, _, _ in arm.chain],
                roll=roll_deg,
                step=STEP_DEG,
                cell=CELL_M,
            )
        ).encode()
    ).hexdigest()[:16]


def _build(arm, roll_deg, key):
    path = CACHE_DIR / f"{key}.npz"
    rmap = None
    if path.is_file():
        try:
            rmap = ReachMap.load(arm, path)
        except (OSError, ValueError, KeyError):
            rmap = None
    if rmap is None:
        rmap = ReachMap(arm, np.deg2rad(roll_deg))
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            rmap.save(path)
        except OSError:
            pass
    with _LOCK:
        if len(_CACHE) > 8:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = rmap
        _BUILDING.pop(key, None)
    return rmap


def reach_map(arm, roll_rad, wait=True):
    """Cached map for this arm geometry at a wrist roll (1 degree bins).

    With ``wait=False`` a missing map is built in a background thread and
    None is returned, so the first click never blocks on the ~10 s build.
    """
    roll_deg = int(round(float(np.rad2deg(roll_rad))))
    key = _key(arm, roll_deg)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        building = _BUILDING.get(key)
        if building is None:
            building = _BUILDING[key] = threading.Thread(
                target=_build, args=(arm, roll_deg, key), daemon=True
            )
            building.start()
    if not wait:
        return None
    building.join()
    return _CACHE.get(key)


def check_reach(arm, point, roll_rad, config=None, label="Selected target"):
    """Raise a precise error when a point is provably outside the arm's reach.

    The map is built for a fixed wrist roll, so it is only applied while the
    roll joint is locked (the normal reach mode); otherwise it stays silent.
    """
    if not hasattr(arm, "chain") or os.environ.get("DESKBUDDY_REACH_MAP") == "0":
        return
    if config is not None and "wrist_roll" not in config.get("locked_joints", []):
        return
    try:
        rmap = reach_map(arm, roll_rad, wait=False)
    except ValueError:
        return
    if rmap is not None and rmap.outside(point):
        gap = rmap.distance_outside(point)
        raise ValueError(
            f"{label} is outside the arm's reach"
            + (f" by at least {gap * 1000:.0f} mm" if gap > 0.0005 else "")
            + " (link lengths and joint ranges)"
        )


def save_report(path, arm, roll_rad, floor_z=0.0):
    rmap = reach_map(arm, roll_rad)
    low, high = rmap.bounds(floor_z)
    report = dict(rmap.summary(), bounding_box_m=dict(min=low, max=high))
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def full_reach_bounds(arm, config, roll_rad=None):
    """Largest motion box: everything the tool tip can reach above the table.

    XY and the top come from the certified reach envelope (link lengths and
    measured joint ranges); the floor is the table plus link clearance, as the
    paper calibration used. Points inside the box can still be out of reach;
    check_reach reports those precisely.
    """
    if roll_rad is None:
        roll_rad = np.deg2rad(config.get("validated_wrist_roll_deg") or 0.0)
    rmap = reach_map(arm, roll_rad)
    floor = float(config["table_z_m"]) + float(config["link_clearance_m"]) + 0.002
    low, high = rmap.bounds(floor)
    low[2] = floor
    return [round(v, 4) for v in low], [round(v, 4) for v in high]
