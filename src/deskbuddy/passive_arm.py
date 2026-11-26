"""Calibrated state access only. This adapter has no motion-command method."""

import time
from pathlib import Path

import numpy as np

from .kinematics import JOINTS


class PassiveArm:
    def __init__(self, config):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.config = config
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=config["robot_port"],
                id=config["robot_id"],
                use_degrees=True,
                calibration_dir=Path(config["motor_calibration_dir"]),
            )
        )
        self.bus = self.robot.bus
        self.telemetry = {}
        self.last_telemetry = 0.0
        try:
            self.resume()
        except Exception:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=False)
            raise

    def resume(self):
        # Disable blocks actuator writes, not read-only monitoring. The viewer
        # must reacquire feedback after a stopped worker, even while disabled.
        self.bus.connect()
        if not self.bus.is_calibrated:
            self.bus.disconnect(disable_torque=False)
            raise ValueError("Motor calibration does not match the saved file")

    def read(self):
        values = self.bus.sync_read("Present_Position", num_retry=2)
        goals = self.bus.sync_read("Goal_Position", list(JOINTS), num_retry=1)
        torque = self.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        degrees = np.array([values[name] for name in JOINTS])
        model_deg = (
            degrees * np.asarray(self.config["joint_signs"])
            + self.config["joint_offsets_deg"]
        )
        if time.monotonic() - self.last_telemetry > 2.0:
            try:
                voltage = self.bus.sync_read(
                    "Present_Voltage", normalize=False, num_retry=1
                )
                temperature = self.bus.sync_read(
                    "Present_Temperature", normalize=False, num_retry=1
                )
                self.telemetry = {
                    "voltage_v": {j: float(v) / 10 for j, v in voltage.items()},
                    "temperature_c": temperature,
                    "telemetry_unix_s": time.time(),
                }
            except (OSError, RuntimeError, ValueError) as exc:
                self.telemetry["telemetry_error"] = str(exc)
            self.last_telemetry = time.monotonic()
        return {
            "motor_degrees": dict(zip(JOINTS, degrees.tolist())),
            "model_degrees": dict(zip(JOINTS, model_deg.tolist())),
            "gripper_percent": values["gripper"],
            "torque_enabled": torque,
            "q_model_rad": np.deg2rad(model_deg).tolist(),
            "q_goal_model_rad": np.deg2rad(
                np.array([goals[j] for j in JOINTS])
                * np.asarray(self.config["joint_signs"])
                + self.config["joint_offsets_deg"]
            ).tolist(),
            "model_mapping_verified": self.config["joint_mapping_verified"],
            **self.telemetry,
        }

    def close(self):
        if self.bus.is_connected:
            self.bus.disconnect(disable_torque=False)
