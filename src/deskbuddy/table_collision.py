"""SO-101 collision-body support distances against the physical fitted table plane.

For a plane, minimum distance over a mesh equals minimum over its convex hull.
Unlike inflated joint origins, this tests the actual URDF collision geometry.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def origin_transform(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return transform


@lru_cache(maxsize=1)
def bodies():
    root = ET.parse(
        files("deskbuddy").joinpath("models/so101_new_calib.urdf")
    ).getroot()
    with np.load(files("deskbuddy").joinpath("models/collision_hulls.npz")) as archive:
        meshes = {k: archive[k] for k in archive.files}
    result = {}
    for link in root.findall("link"):
        vertices = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            t = origin_transform(collision.find("origin"))
            v = meshes[Path(mesh.get("filename")).stem]
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            vertices.append((v * scale) @ t[:3, :3].T + t[:3, 3])
        if vertices:
            result[link.get("name")] = np.concatenate(vertices)
    children = {
        j.get("name"): j.find("child").get("link") for j in root.findall("joint")
    }
    jaw = root.find("joint[@name='gripper']")
    return result, children, origin_transform(jaw.find("origin"))


def physical_plane(config):
    from .grasp_geometry import table_normal
    from .test_bed import corners

    normal = table_normal(config)
    bed = corners(config, inset=False)
    anchor = np.array([0.0, 0.0, config["table_z_m"]]) if bed is None else bed[0]
    return normal, -float(normal @ anchor)


def transformed_bodies(arm, q, include_jaw=True):
    geometry, children, jaw_origin = bodies()
    t, index = np.eye(4), 0
    result = []
    for name, origin, axis, bounds in arm.chain:
        t = t @ origin
        if bounds is not None:
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
            t = t @ motion
            index += 1
        child = children[name]
        # Base/pan mount is intentionally attached to the table, not a free body.
        if child in geometry and index >= 2:
            result.append((child, geometry[child], t.copy()))
        if name == "wrist_roll" and include_jaw:
            # Conservative envelope through the jaw's URDF opening range. Plane
            # support is evaluated at 5-degree intervals, with a 0.1 mm allowance.
            vertices = geometry["moving_jaw_so101_v1_link"]
            for angle in np.linspace(np.deg2rad(-10), np.deg2rad(100), 23):
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
                result.append(("moving_jaw", vertices, t @ jaw_origin @ motion))
    return result


@lru_cache(maxsize=1)
def support_geometry():
    """Exact plane-support hulls, including the same sampled jaw envelope."""
    from scipy.spatial import ConvexHull

    geometry, _, jaw_origin = bodies()
    reduced = {}
    for name, vertices in geometry.items():
        reduced[name] = vertices[ConvexHull(vertices).vertices]
    vertices = reduced["moving_jaw_so101_v1_link"]
    envelope = []
    for angle in np.linspace(np.deg2rad(-10), np.deg2rad(100), 23):
        rotation = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        envelope.append(
            (vertices @ rotation.T) @ jaw_origin[:3, :3].T + jaw_origin[:3, 3]
        )
    envelope = np.concatenate(envelope)
    return reduced, envelope[ConvexHull(envelope).vertices]


def plane_support_bodies(arm, q):
    reduced, jaw_envelope = support_geometry()
    # All 23 opening samples share the wrist frame. Their convex hull has
    # exactly the same minimum dot product for every possible table normal.
    for name, _, pose in transformed_bodies(arm, q, include_jaw=False):
        yield name, reduced[name], pose
        if name == "gripper_link":
            # gripper_link is the wrist-roll child in the commissioned URDF.
            yield "moving_jaw", jaw_envelope, pose


def clearances(arm, q, config):
    if not hasattr(arm, "chain"):
        from .test_bed import table_clearance

        values = table_clearance(arm.frames(q)[0][2:], config)
        return [(f"reference {i}", float(v)) for i, v in enumerate(values)]
    normal, offset = physical_plane(config)
    result = []
    for name, vertices, t in plane_support_bodies(arm, q):
        distance = float(
            np.min(vertices @ (t[:3, :3].T @ normal)) + normal @ t[:3, 3] + offset
        )
        # Body clearance is measured from its surface, without stacking the
        # historical 3 mm plane offset and 15 mm joint-origin margin.
        margin = (
            0.0005
            if config.get("grasp_contact_mode")
            and name in ("gripper_link", "moving_jaw")
            else 0.001
        )
        if name == "moving_jaw":
            margin += 0.0001
        result.append((name, distance - margin))
    tip = arm.fk(q)
    result.append(("calibrated jaw tip", float(tip @ normal + offset)))
    grouped = {}
    for name, distance in result:
        grouped[name] = min(distance, grouped.get(name, float("inf")))
    return list(grouped.items())


def minimum_clearance(arm, q, config):
    return min(v for _, v in clearances(arm, q, config))


def rejection(arm, q, config, phase="Planned pose"):
    name, gap = min(clearances(arm, q, config), key=lambda item: item[1])
    return (
        f"{phase}: {name} table clearance {gap * 1000:.1f} mm (collision-body estimate)"
    )


def path_free(arm, path, config):
    """Adaptive support-distance bound; no jump between sparse samples is assumed safe."""
    cache = {}

    def gap(q):
        key = tuple(q)
        if key not in cache:
            cache[key] = minimum_clearance(arm, q, config)
        return cache[key]

    # Sum of URDF offsets plus a body extent bounds distance from any joint axis.
    radius = sum(np.linalg.norm(origin[:3, 3]) for _, origin, _, _ in arm.chain) + 0.25

    def edge(a, b, depth=0):
        left, right = gap(a), gap(b)
        if min(left, right) < -1e-7:
            return False
        motion_bound = radius * np.sum(abs(b - a))
        if min(left, right) >= motion_bound:
            return True
        if depth >= 16:
            return False  # cannot certify a grazing segment
        middle = (a + b) / 2
        return edge(a, middle, depth + 1) and edge(middle, b, depth + 1)

    return all(edge(a, b) for a, b in zip(path[:-1], path[1:]))
