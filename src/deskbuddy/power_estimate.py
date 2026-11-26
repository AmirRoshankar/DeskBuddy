"""Quasi-static motor input-power scenario, not measured electrical telemetry."""

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from importlib.resources import files

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import Arm


@dataclass(frozen=True)
class MotorAssumptions:
    voltage_v: float = 5.4  # Last observed rail voltage; scenario input, not live.
    shaft_torque_constant_nm_a: float = 7.8 * 0.0980665
    winding_resistance_ohm: float = 2.5
    electronics_idle_a: float = 0.006
    no_load_running_a: float = 0.150
    no_load_speed_rad_s: float = np.deg2rad(60) / 0.192
    mechanical_efficiency: float = 0.6  # Unmeasured assumption.
    payload_kg: float = 0.0


class QuasiStaticPower:
    def __init__(self, config, assumptions=MotorAssumptions()):
        if (
            assumptions.voltage_v <= 0
            or assumptions.winding_resistance_ohm <= 0
            or assumptions.shaft_torque_constant_nm_a <= 0
            or not 0 < assumptions.mechanical_efficiency <= 1
            or assumptions.payload_kg < 0
        ):
            raise ValueError("Invalid power scenario assumptions")
        self.arm = Arm(
            config.get("model_joint_limits_deg"), config.get("tool_offset_m")
        )
        self.assumptions = assumptions
        root = ET.parse(
            files("deskbuddy").joinpath("models/so101_new_calib.urdf")
        ).getroot()
        links = {link.get("name"): link for link in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.bodies = []
        for name, *_ in self.arm.chain:
            child = self.joints[name].find("child").get("link")
            inertial = links[child].find("inertial")
            self.bodies.append(
                (
                    float(inertial.find("mass").get("value")),
                    np.fromstring(inertial.find("origin").get("xyz"), sep=" "),
                )
            )
        jaw = links["moving_jaw_so101_v1_link"].find("inertial")
        origin = self.joints["gripper"].find("origin")
        self.jaw_transform = np.eye(4)
        self.jaw_transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy"), sep=" ")
        ).as_matrix()
        self.jaw_transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        self.jaw_mass = float(jaw.find("mass").get("value"))
        self.jaw_com = np.fromstring(jaw.find("origin").get("xyz"), sep=" ")

    def potential_j(self, q):
        transform, index, potential = np.eye(4), 0, 0.0
        gripper = None
        for (name, origin, axis, bounds), (mass, com) in zip(
            self.arm.chain, self.bodies
        ):
            transform = transform @ origin
            if bounds is not None:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                transform = transform @ motion
                index += 1
            point = transform[:3, :3] @ com + transform[:3, 3]
            potential += mass * 9.80665 * point[2]
            if name == "wrist_roll":
                gripper = transform.copy()
        # Jaw is closed/fixed in this experiment; its motor is not powered.
        jaw = gripper @ self.jaw_transform
        potential += (
            self.jaw_mass * 9.80665 * (jaw[:3, :3] @ self.jaw_com + jaw[:3, 3])[2]
        )
        return potential + self.assumptions.payload_kg * 9.80665 * self.arm.fk(q)[2]

    def gravity_torque_nm(self, q):
        epsilon = 1e-5
        result = []
        for direction in np.eye(5) * epsilon:
            result.append(
                (self.potential_j(q + direction) - self.potential_j(q - direction))
                / (2 * epsilon)
            )
        return np.array(result)

    def power_w(self, q, qdot):
        a = self.assumptions
        torque = self.gravity_torque_nm(q)
        coil_current = np.abs(torque) / a.shaft_torque_constant_nm_a
        copper = a.winding_resistance_ohm * coil_current**2
        positive_mechanical = np.maximum(torque * qdot, 0) / a.mechanical_efficiency
        moving_loss = (
            a.voltage_v
            * max(0.0, a.no_load_running_a - a.electronics_idle_a)
            * np.minimum(np.abs(qdot) / a.no_load_speed_rad_s, 1)
        )
        per_joint = (
            a.voltage_v * a.electronics_idle_a
            + copper
            + positive_mechanical
            + moving_loss
        )
        return per_joint, torque

    def evaluate(self, plan, max_samples=61):
        indices = np.unique(
            np.linspace(
                0, len(plan.time_s) - 1, min(max_samples, len(plan.time_s))
            ).astype(int)
        )
        velocity = np.gradient(
            plan.joints_rad,
            plan.time_s,
            axis=0,
            edge_order=2 if len(plan.time_s) > 2 else 1,
        )
        time = plan.time_s[indices]
        powers, torques = zip(
            *(self.power_w(plan.joints_rad[i], velocity[i]) for i in indices)
        )
        powers = np.array(powers)
        total = powers.sum(1)
        energy = float(np.trapezoid(total, time))
        return dict(
            estimated=True,
            model="quasi_static_gravity_plus_copper_and_running_losses",
            duration_s=float(time[-1] - time[0]),
            energy_j=energy,
            energy_wh=energy / 3600,
            mean_power_w=energy / (time[-1] - time[0]),
            peak_power_w=float(total.max()),
            per_joint_energy_j=np.trapezoid(powers, time, axis=0).tolist(),
            max_abs_gravity_torque_nm=np.abs(torques).max(0).tolist(),
            time_s=time.tolist(),
            total_power_w=total.tolist(),
            per_joint_power_w=powers.tolist(),
            assumptions=asdict(self.assumptions),
            limitations="Not measured. Ignores inertial/Coriolis torques, servo PWM details, friction characterization, controller/camera/PC draw, payload by default, and post-reach holding. No regenerative credit. Vertical base gravity assumed.",
        )
