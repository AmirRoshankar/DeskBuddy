"""Non-adjacent URDF convex collision parts, not inflated joint-center lines.

Compound links keep their separate collision parts. Hulls still approximate mesh
concavities; a positive inscribed overlap is a model prediction, not contact sensing.
"""

import itertools
import xml.etree.ElementTree as ET
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .kinematics import axis_rotation
from .table_collision import origin_transform


@lru_cache(maxsize=1)
def model():
    root = ET.parse(
        files("deskbuddy").joinpath("models/so101_new_calib.urdf")
    ).getroot()
    with np.load(files("deskbuddy").joinpath("models/collision_hulls.npz")) as archive:
        meshes = {k: archive[k] for k in archive.files}
    parts = {}
    for link in root.findall("link"):
        components = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            vertices = meshes[Path(mesh.get("filename")).stem]
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            t = origin_transform(collision.find("origin"))
            vertices = (vertices * scale) @ t[:3, :3].T + t[:3, 3]
            hull = ConvexHull(vertices)
            equations = np.unique(np.round(hull.equations, 10), axis=0)
            components.append((vertices[hull.vertices], equations))
        if components:
            parts[link.get("name")] = components
    children = {
        j.get("name"): j.find("child").get("link") for j in root.findall("joint")
    }
    allowed = {
        frozenset((j.find("parent").get("link"), j.find("child").get("link")))
        for j in root.findall("joint")
    }
    allowed.add(frozenset(("wrist_link", "moving_jaw_so101_v1_link")))
    jaw = root.find("joint[@name='gripper']")
    return parts, children, allowed, origin_transform(jaw.find("origin"))


class LazyPart(dict):
    """World collision part; face planes are transformed only when a face test needs them."""

    def __missing__(self, key):
        if key not in ("normals", "offsets"):
            raise KeyError(key)
        t, equations = self["_transform"], self["_equations"]
        normals = equations[:, :3] @ t[:3, :3].T
        self["normals"] = normals
        self["offsets"] = equations[:, 3] - normals @ t[:3, 3]
        return self[key]


def world_parts(arm, q, jaw_deg=0.0):
    parts, children, allowed, jaw_origin = model()
    transforms = {"base_link": np.eye(4)}
    t = np.eye(4)
    index = 0
    for name, origin, axis, bounds in arm.chain:
        t = t @ origin
        if bounds is not None:
            motion = np.eye(4)
            motion[:3, :3] = axis_rotation(axis, q[index])
            t = t @ motion
            index += 1
        transforms[children[name]] = t.copy()
        if name == "wrist_roll":
            # Nominal closed jaw, including Home's requested closed end pose.
            opening = np.eye(4)
            # Nominal 0-100 % opening ~ 0-100 URDF degrees about the jaw axis
            # (same mapping as can_discard.jaw_geometry).
            opening[:3, :3] = axis_rotation(
                np.array([0.0, 0.0, 1.0]), np.deg2rad(jaw_deg)
            )
            transforms["moving_jaw_so101_v1_link"] = t @ jaw_origin @ opening
    result = []
    for name, t in transforms.items():
        for index, (vertices, equations) in enumerate(parts.get(name, [])):
            world = vertices @ t[:3, :3].T + t[:3, 3]
            result.append(
                LazyPart(
                    name=name,
                    part=index,
                    vertices=world,
                    low=world.min(0),
                    high=world.max(0),
                    _equations=equations,
                    _transform=t,
                )
            )
    return result, allowed


def _closest_simplex(simplex):
    """Point of the simplex nearest the origin, and the sub-simplex supporting it."""
    points = np.asarray(simplex)
    count = len(points)
    best = None
    # Enumerate faces of the (at most 4-point) simplex: exact, tiny and robust.
    for size in range(1, count + 1):
        for subset in itertools.combinations(range(count), size):
            p = points[list(subset)]
            if size == 1:
                weights = np.ones(1)
            else:
                edges = (p[1:] - p[0]).T
                gram = edges.T @ edges
                try:
                    coefficients = np.linalg.solve(gram, -edges.T @ p[0])
                except np.linalg.LinAlgError:
                    continue
                weights = np.r_[1 - coefficients.sum(), coefficients]
                if (weights < -1e-12).any():
                    continue
            closest = weights @ p
            distance = float(closest @ closest)
            if best is None or distance < best[0] - 1e-18:
                best = (distance, closest, list(subset))
    return best[1], [simplex[i] for i in best[2]]


def separated(a, b, iterations=48):
    """Exact separation certificate (GJK search); False means "not proven".

    A returned True is only issued after checking a plane that strictly
    separates both vertex sets, so search errors can only cost time.
    """
    va, vb = a["vertices"], b["vertices"]
    direction = vb.mean(0) - va.mean(0)
    if direction @ direction < 1e-18:
        return False
    # Minkowski difference B - A; the origin is outside iff the bodies are apart.
    simplex = []
    for _ in range(iterations):
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return False
        d = direction / norm
        projected_a, projected_b = va @ d, vb @ d
        # Strictly separating plane along d: certify and stop.
        if projected_a.max() < projected_b.min():
            return True
        support = vb[np.argmin(projected_b)] - va[np.argmax(projected_a)]
        if any(np.allclose(support, s, atol=1e-12) for s in simplex):
            return False
        simplex.append(support)
        closest, simplex = _closest_simplex(simplex)
        if len(simplex) == 4 or closest @ closest < 1e-18:
            return False
        direction = closest
    return False


def overlap_radius(a, b, tolerance=0.0005):
    # An AABB or face-axis separation proves no intersection; only ambiguous
    # cases reach the tiny 4-variable convex feasibility problem.
    if np.any(
        np.minimum(a["high"], b["high"]) - np.maximum(a["low"], b["low"])
        <= 2 * tolerance
    ):
        return 0.0
    # Cheap exact separating plane first: the face-axis products below cost
    # thousands of faces times thousands of vertices per pair.
    if separated(a, b):
        return 0.0
    for first, second in ((a, b), (b, a)):
        minimum = (first["normals"] @ second["vertices"].T).min(1) + first["offsets"]
        if np.any(minimum >= -2 * tolerance):
            return 0.0
    normals = np.vstack([a["normals"], b["normals"]])
    offsets = np.r_[a["offsets"], b["offsets"]]
    fit = linprog(
        [0, 0, 0, -1],
        A_ub=np.c_[normals, np.ones(len(normals))],
        b_ub=-offsets,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if not fit.success:
        raise ValueError("Self collision geometry check inconclusive")
    return max(0.0, float(fit.x[3]))


def collisions(arm, q, tolerance=0.0005):
    parts, allowed = world_parts(arm, q)
    result = []
    for a, b in itertools.combinations(parts, 2):
        if a["name"] == b["name"] or frozenset((a["name"], b["name"])) in allowed:
            continue
        radius = overlap_radius(a, b, tolerance)
        if radius > tolerance:
            result.append(
                dict(first=a["name"], second=b["name"], overlap_radius_m=radius)
            )
    return result


def check(arm, q):
    hits = collisions(arm, q)
    if hits:
        hit = max(hits, key=lambda item: item["overlap_radius_m"])
        raise ValueError(
            f"Self collision: {hit['first']} / {hit['second']} (URDF body overlap {hit['overlap_radius_m'] * 1000:.1f} mm radius)"
        )
