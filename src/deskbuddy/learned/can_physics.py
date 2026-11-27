"""Headless MuJoCo can-rim pick/transfer/drop teacher, never robot hardware.

Rigid cylinder plus a raised lip, finite-force parallel fingers, gravity and
Coulomb contacts. This intentionally does not simulate an entire SO-101 or a
deformable aluminum can; successful labels are not physical-robot evidence.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Default toy rim fingers (half sizes x, y, z in metres) and per-finger travel.
RIM_FINGER = (0.0015, 0.004, 0.012)
RIM_TRAVEL = 0.02
# Body-grasp fingers: an assumed SO-101-like jaw pad, 4 x 20 x 40 mm, with
# enough travel to open around a 74 mm can. Not a measured jaw model.
BODY_FINGER = (0.002, 0.01, 0.02)
BODY_TRAVEL = 0.045


def scene_xml(
    diameter,
    height,
    mass=0.02,
    friction=0.8,
    force=4.0,
    finger=RIM_FINGER,
    travel=RIM_TRAVEL,
    lid_diameter=None,
):
    """Toy can scene. ``lid_diameter`` adds a necked top (real 211/202 cans
    neck from the 66 mm body to a ~54 mm lid over ~19 mm); None keeps the
    original straight-walled toy for reproducing earlier datasets."""
    fx, fy, fz = finger
    if lid_diameter is not None:
        return _necked_xml(
            diameter, height, lid_diameter, mass, friction, force, fx, fy, fz, travel
        )
    r = diameter / 2
    # A recessed solid lid/body and a rolled raised lip leave room for an
    # inside/outside pinch. The 6 mm recess is a deliberately explicit toy prior.
    ring = []
    for angle in np.arange(24) * (2 * np.pi / 24):
        other = angle + 2 * np.pi / 24
        a = [(r - 0.001) * np.cos(angle), (r - 0.001) * np.sin(angle), height - 0.001]
        b = [(r - 0.001) * np.cos(other), (r - 0.001) * np.sin(other), height - 0.001]
        ring.append(
            f'<geom type="capsule" fromto="{" ".join(map(str, a + b))}" size=".001" mass=".00005"/>'
        )
    return f'''<mujoco model="toy_can_rim">
      <option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default><geom friction="{friction} .005 .0001" condim="4" solref=".006 1"/>
      <joint damping="1"/></default>
      <worldbody>
        <geom name="table" type="plane" size="1 1 .01"/>
        <body name="can" pos="0 0 .001"><freejoint/>
          <geom name="body" type="cylinder" size="{r - 0.001} {(height - 0.006) / 2}" pos="0 0 {(height - 0.006) / 2}" mass="{max(0.003, mass - 0.0012)}"/>
          {"".join(ring)}
        </body>
        <body name="target" mocap="true" pos="{r} 0 {height + 0.04}"/>
        <body name="hand" pos="{r} 0 {height + 0.04}"><freejoint name="hand_free"/>
          <inertial pos="0 0 0" mass=".05" diaginertia=".0001 .0001 .0001"/>
          <body pos="-{fx} 0 0"><joint name="left" type="slide" axis="-1 0 0" range="0 {travel}"/>
            <geom type="box" size="{fx} {fy} {fz}" pos="0 0 -{fz}" mass=".025"/>
          </body>
          <body pos="{fx} 0 0"><joint name="right" type="slide" axis="1 0 0" range="0 {travel}"/>
            <geom type="box" size="{fx} {fy} {fz}" pos="0 0 -{fz}" mass=".025"/>
          </body>
        </body>
      </worldbody>
      <equality><weld body1="hand" body2="target" relpose="0 0 0 1 0 0 0" solref=".005 1"/></equality>
      <actuator>
        <position joint="left" kp="5000" kv="10" forcerange="-{force} {force}"/>
        <position joint="right" kp="5000" kv="10" forcerange="-{force} {force}"/>
      </actuator>
    </mujoco>'''


NECK_HEIGHT = 0.019


def _ring(radius, z, size):
    parts = []
    for angle in np.arange(32) * (2 * np.pi / 32):
        other = angle + 2 * np.pi / 32
        a = [radius * np.cos(angle), radius * np.sin(angle), z]
        b = [radius * np.cos(other), radius * np.sin(other), z]
        parts.append(
            f'<geom type="capsule" fromto="{" ".join(map(str, a + b))}" size="{size}" mass=".00005"/>'
        )
    return "".join(parts)


def _necked_xml(diameter, height, lid, mass, friction, force, fx, fy, fz, travel):
    """Body cylinder, stepped conical neck, recessed panel and a seam ring.

    Seam: a 2.6 mm thick rolled ring at the lid radius, its top at the can
    height; panel 6 mm below. Dimensions follow 211/202 can specs loosely.
    """
    body_r, lid_r = diameter / 2, lid / 2
    neck_start = height - NECK_HEIGHT
    panel = height - 0.006
    geoms = [
        f'<geom name="body" type="cylinder" size="{body_r - 0.0005} {neck_start / 2}" '
        f'pos="0 0 {neck_start / 2}" mass="{max(0.003, mass - 0.002)}"/>'
    ]
    steps = 5
    for index in range(steps):
        z0 = neck_start + (panel - neck_start) * index / steps
        z1 = neck_start + (panel - neck_start) * (index + 1) / steps
        radius = body_r + (lid_r - body_r) * (index + 0.5) / steps
        geoms.append(
            f'<geom type="cylinder" size="{radius - 0.0005} {(z1 - z0) / 2}" '
            f'pos="0 0 {(z0 + z1) / 2}" mass=".0003"/>'
        )
    seam = _ring(lid_r - 0.0013, height - 0.0013, 0.0013)
    return f'''<mujoco model="toy_necked_can">
      <option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default><geom friction="{friction} .005 .0001" condim="4" solref=".006 1"/>
      <joint damping="1"/></default>
      <worldbody>
        <geom name="table" type="plane" size="1 1 .01"/>
        <body name="can" pos="0 0 .001"><freejoint/>
          {"".join(geoms)}
          {seam}
        </body>
        <body name="target" mocap="true" pos="{lid_r} 0 {height + 0.04}"/>
        <body name="hand" pos="{lid_r} 0 {height + 0.04}"><freejoint name="hand_free"/>
          <inertial pos="0 0 0" mass=".05" diaginertia=".0001 .0001 .0001"/>
          <body pos="-{fx} 0 0"><joint name="left" type="slide" axis="-1 0 0" range="0 {travel}"/>
            <geom type="box" size="{fx} {fy} {fz}" pos="0 0 -{fz}" mass=".025"/>
          </body>
          <body pos="{fx} 0 0"><joint name="right" type="slide" axis="1 0 0" range="0 {travel}"/>
            <geom type="box" size="{fx} {fy} {fz}" pos="0 0 -{fz}" mass=".025"/>
          </body>
        </body>
      </worldbody>
      <equality><weld body1="hand" body2="target" relpose="0 0 0 1 0 0 0" solref=".005 1"/></equality>
      <actuator>
        <position joint="left" kp="5000" kv="10" forcerange="-{force} {force}"/>
        <position joint="right" kp="5000" kv="10" forcerange="-{force} {force}"/>
      </actuator>
    </mujoco>'''


def rollout(
    diameter,
    height,
    offset,
    quaternion,
    opening,
    mass=0.02,
    friction=0.8,
    force=4.0,
    finger=RIM_FINGER,
    travel=RIM_TRAVEL,
    approach=0.04,
    lid_diameter=None,
):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        scene_xml(diameter, height, mass, friction, force, finger, travel, lid_diameter)
    )
    data = mujoco.MjData(model)
    can_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
    for joint in ("left", "right"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[joint_id]] = opening / 2
    quat = np.asarray(quaternion, float)
    quat = quat / max(np.linalg.norm(quat), 1e-9)
    data.mocap_quat[0] = quat
    target = np.asarray(offset, float)
    data.mocap_pos[0] = target + [0, 0, approach]
    hand_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
    hand_address = model.jnt_qposadr[hand_joint]
    data.qpos[hand_address : hand_address + 3] = data.mocap_pos[0]
    data.qpos[hand_address + 3 : hand_address + 7] = quat
    data.ctrl[:] = opening / 2
    mujoco.mj_forward(model, data)
    max_height = 0.0
    retained_height = 0.0
    initial_xy = None
    contacts = 0

    def phase(a, b, seconds, gap):
        nonlocal max_height, contacts
        steps = int(seconds / model.opt.timestep)
        for index in range(steps):
            fraction = (index + 1) / steps
            smooth = fraction * fraction * (3 - 2 * fraction)
            data.mocap_pos[0] = a + (b - a) * smooth
            data.ctrl[:] = gap / 2
            mujoco.mj_step(model, data)
            max_height = max(max_height, float(data.xpos[can_id, 2]))
            contacts = max(contacts, int(data.ncon))

    above = target + [0, 0, approach]
    phase(above, above, 0.1, opening)
    initial_xy = data.xpos[can_id, :2].copy()
    phase(above, target, 0.3, opening)
    phase(target, target, 0.25, 0.0008)
    lifted = target + [0, 0, 0.06]
    phase(target, lifted, 0.4, 0.0008)
    phase(lifted, lifted, 0.2, 0.0008)
    retained_height = float(data.xpos[can_id, 2])

    def bottom():
        rotation = data.xmat[can_id].reshape(3, 3)
        center = data.xpos[can_id] + rotation @ np.array([0, 0, height / 2])
        return float(
            center[2]
            - abs(rotation[2, 2]) * height / 2
            - diameter / 2 * np.linalg.norm(rotation[2, :2])
        )

    retained_clearance = bottom()
    pickup = retained_clearance > 0.025
    destination = lifted + [0.12, 0, 0]
    phase(lifted, destination, 0.35, 0.0008)
    transfer_clearance = bottom()
    release = max(0.025, opening)
    phase(destination, destination, 0.15, release)
    phase(destination, destination, 0.5, release)
    final = data.xpos[can_id].copy()
    dropped = bool(
        abs(final[0] - 0.12) < 0.05 and abs(final[1]) < 0.05 and abs(bottom()) < 0.005
    )
    return dict(
        pickup=bool(pickup),
        transfer=bool(transfer_clearance > 0.02),
        drop_in_zone=dropped,
        success=bool(pickup and transfer_clearance > 0.02 and dropped),
        retained_bottom_clearance_m=retained_clearance,
        transfer_bottom_clearance_m=transfer_clearance,
        retained_base_height_m=retained_height,
        max_base_height_m=max_height,
        final_base_m=final.tolist(),
        initial_base_xy_m=initial_xy.tolist(),
        contacts_max=contacts,
        physics=True,
        gravity_m_s2=9.81,
    )


def body_rollout(
    diameter, height, yaw, mass=0.02, friction=0.8, force=4.0, lid_diameter=None
):
    """Numerical body-grasp baseline: centered, closing across the diameter.

    Mirrors the live body grasp (opening = diameter + 6 mm, contact centred
    near 55% height) with the assumed BODY_FINGER pads, descending from above
    the can top so the fingers never start inside it.
    """
    tip_z = max(0.002, 0.55 * height - BODY_FINGER[2])
    quat = [0, np.cos(yaw / 2), np.sin(yaw / 2), 0]
    result = rollout(
        diameter,
        height,
        [0.0, 0.0, tip_z],
        quat,
        min(diameter + 0.006, 2 * BODY_TRAVEL - 1e-4),
        mass,
        friction,
        force,
        finger=BODY_FINGER,
        travel=BODY_TRAVEL,
        approach=height - tip_z + 0.03,
        lid_diameter=lid_diameter,
    )
    result["grasp"] = "numerical_body_baseline"
    return result


def generate(folder, objects=500, attempts=4, seed=20261126, focused=False):
    import mujoco

    from .data import check_budget

    folder = Path(folder)
    if (folder / "dataset.npz").exists():
        raise ValueError("Choose a fresh physics dataset folder")
    folder.mkdir(parents=True, exist_ok=True)
    check_budget(folder, 50_000_000)
    rng = np.random.default_rng(seed)
    contexts, targets, groups, outcomes, parameters = [], [], [], [], []
    began = time.perf_counter()
    for index in range(objects):
        diameter = rng.uniform(0.06, 0.074)
        height = rng.uniform(0.115, 0.13)
        mass = rng.uniform(0.015, 0.4)
        friction = rng.uniform(0.35, 1.1)
        for attempt in range(attempts):
            angle = rng.uniform(-np.pi, np.pi)
            yaw = (angle + np.pi / 2) % np.pi - np.pi / 2
            radius = diameter / 2 + rng.uniform(
                *((-0.0005, 0.0015) if focused else (-0.002, 0.002))
            )
            offset = np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    height
                    - rng.uniform(*((0.0024, 0.0038) if focused else (0.0005, 0.003))),
                ]
            )
            quat = [0, np.cos(yaw / 2), np.sin(yaw / 2), 0]
            opening = rng.uniform(0.006, 0.012)
            result = rollout(diameter, height, offset, quat, opening, mass, friction)
            contexts.append(
                [
                    diameter / 0.05,
                    diameter / 0.05,
                    height / 0.05,
                    np.cos(angle),
                    np.sin(angle),
                ]
            )
            targets.append([*(offset / 0.05), *quat, opening / 0.05])
            groups.append(index)
            outcomes.append(result)
            parameters.append([mass, friction, 4.0])
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    dict(
                        objects=index + 1,
                        seconds=round(time.perf_counter() - began, 1),
                        successes=sum(o["success"] for o in outcomes),
                    )
                ),
                flush=True,
            )
    valid = np.array([o["success"] for o in outcomes])
    np.savez_compressed(
        folder / "all-rollouts.npz",
        contexts=np.float32(contexts),
        targets=np.float32(targets),
        object_id=groups,
        success=valid,
        physics_parameters=np.float32(parameters),
    )
    np.savez_compressed(
        folder / "dataset.npz",
        contexts=np.float32(contexts)[valid],
        targets=np.float32(targets)[valid],
        object_id=np.asarray(groups)[valid],
    )
    report = dict(
        engine=f"MuJoCo {mujoco.__version__}",
        gravity_m_s2=9.81,
        objects=objects,
        rollouts=len(outcomes),
        successful_rollouts=int(valid.sum()),
        success_rate=float(valid.mean()),
        elapsed_s=time.perf_counter() - began,
        seed=seed,
        proposal_sampling="pilot-informed rim depth/radius"
        if focused
        else "broad rim perturbations",
        motor_io=False,
        full_robot_simulated=False,
        assumptions=dict(
            rigid_can=True,
            lid_recess_m=0.006,
            lip_radius_m=0.001,
            finger_width_m=0.003,
            finger_depth_m=0.008,
            finger_length_m=0.024,
            force_limit_per_finger_n=4,
            mass_kg=[0.015, 0.4],
            friction=[0.35, 1.1],
        ),
        limitations=[
            "Mocap hand; SO-101 dynamics/reachability not simulated.",
            "Rigid toy raised lip/recess, not measured can mesh or deformable aluminum.",
            "No claim of real-robot success; all failures retained in all-rollouts.npz.",
        ],
    )
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--objects", type=int, default=500)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--focused",
        action="store_true",
        help="Sample deeper lip contacts suggested by the separate pilot",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            generate(args.folder, args.objects, args.attempts, focused=args.focused)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
