#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
import mujoco
import mujoco.viewer
# from lerobot.cameras.utils import make_cameras_from_configs this will call the built-in function to make cameras but we use simulated cameras.
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.simulation.norm_denorm import RobotJointController

from ..robot import Robot
from .config_sim_101 import Sim101Config

logger = logging.getLogger(__name__)


class Sim101(Robot):
    """
    Simulated SO-101 Follower Arm.
    """

    config_class = Sim101Config
    name = "sim_101"

    def __init__(self, config: Sim101Config, model: mujoco.MjModel = None, data: mujoco.MjData = None):
        super().__init__(config)
        self.config = config
        self.joint_controller = RobotJointController()

        # MuJoCo simulation components
        self.model = model
        self.data = data
        self.viewer = None
        
        self.cameras = {"front":"camera_front", "gripper": "camera_gripper"}

    @classmethod
    def from_sim(cls, config: Sim101Config, model: mujoco.MjModel, data: mujoco.MjData):
        return cls(config, model, data)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.joint_controller.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Load the MuJoCo model if not already loaded
        try:
            if self.model is None:
                # TODO(vikashplus): Make the model path configurable
                self.model = mujoco.MjModel.from_xml_path("/home/bfl3/bfl_works/new_lerobot/lerobot/src/lerobot/simulation/SO101/scene.xml")
                self.data = mujoco.MjData(self.model)
                self.renderer = mujoco.Renderer(self.model, 480, 640) #for getting camera input.
                mujoco.mj_forward(self.model, self.data)

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Wait for the viewer to be ready
            start_time = time.time()
            while not self.viewer.is_running():
                if time.time() - start_time > 5:  # 5-second timeout
                    raise DeviceNotConnectedError("MuJoCo viewer failed to start.")
                time.sleep(0.1)

            logger.info(f"{self} connected.")
        except Exception as e:
            raise DeviceNotConnectedError(f"Failed to connect to MuJoCo simulation: {e}")



    @property
    def is_calibrated(self) -> bool:
        # In simulation, we assume the robot is always calibrated
        return True

    def calibrate(self) -> None:
        logger.info(f"Calibration is not required for the simulated robot {self.name}.")

    def configure(self) -> None:
        logger.info(f"Configuration is not required for the simulated robot {self.name}.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read joint positions from the simulation
        start = time.perf_counter()
        qpos = self.data.qpos[:len(self.joint_controller.joint_names)]
        normalized_qpos = self.joint_controller._normalize_array(qpos)
        
        obs_dict = {f"{name}.pos": val for name, val in zip(self.joint_controller.joint_names, normalized_qpos)}
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            # TODO(vikashplus): Implement image capture from MuJoCo
            # For now, returning a black image
            self.renderer.update_scene(self.data, camera=cam)
            obs_dict[cam_key] = self.renderer.render()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract normalized joint positions from the action dictionary
        normalized_angles = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        # Convert to an ordered array for the joint controller
        normalized_array = np.array([normalized_angles[name] for name in self.joint_controller.joint_names])

        # Unnormalize to get radian values for the simulation
        radian_angles = self.joint_controller._unnormalize_array(normalized_array)

        # Send to MuJoCo
        self.data.ctrl[:len(radian_angles)] = radian_angles
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.viewer.close()
        self.viewer = None
        self.model = None
        self.data = None
        
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
