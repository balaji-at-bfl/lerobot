# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import MotorNormMode

from ..robot import Robot
from .config_so101_sim_follower import SO101SimFollowerConfig

logger = logging.getLogger(__name__)


class SO101SimFollower(Robot):
    """
    Simulated SO-101 Follower Arm using MuJoCo.
    """

    config_class = SO101SimFollowerConfig
    name = "so101_sim_follower"

    def __init__(self, config: SO101SimFollowerConfig):
        super().__init__(config)
        self.config = config
        self._model = None
        self._data = None
        self._viewer = None
        self._is_connected = False

        # Define joint names based on the SO101 model
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.camera_names = list(config.cameras.keys())

        # MuJoCo camera IDs
        self._mujoco_camera_ids = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.camera_names
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # Simulation does not require calibration
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self._model = mujoco.xml.load(str(self.config.mujoco_model_path))
            self._data = mujoco.mju_zero(self._model.nexternal)
            self._data = mujoco.mj_makeData(self._model)
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model from {self.config.mujoco_model_path}: {e}")
            raise

        # Initialize camera IDs
        for cam_name in self.camera_names:
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                logger.warning(f"MuJoCo camera '{cam_name}' not found in model. Skipping.")
            self._mujoco_camera_ids[cam_name] = cam_id

        # Initialize viewer if display_data is true (handled by record/teleoperate scripts)
        # We don't create the viewer here directly, as it's managed by the main script's display_data flag.
        # The main script will call configure() which can then create the viewer.

        self._is_connected = True
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        # No calibration needed for simulation
        logger.info(f"No calibration needed for simulated {self}.")
        pass

    def configure(self) -> None:
        # Reset simulation to initial state
        if self._model and self._data:
            mujoco.mj_resetData(self._model, self._data)
            # Set initial joint positions if desired (e.g., to a rest pose)
            # For now, just reset to default.
        logger.info(f"{self} configured.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Get joint positions
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                # Assuming joint positions are in qpos
                obs_dict[f"{joint_name}.pos"] = self._data.qpos[self._model.jnt_qposadr[joint_id]].item()
            else:
                logger.warning(f"Joint '{joint_name}' not found in MuJoCo model.")
                obs_dict[f"{joint_name}.pos"] = 0.0 # Default value

        # Get camera images
        for cam_name, cam_id in self._mujoco_camera_ids.items():
            if cam_id != -1:
                width = self.config.cameras[cam_name].width
                height = self.config.cameras[cam_name].height
                img = np.zeros((height, width, 3), dtype=np.uint8)
                mujoco.mjr_render(
                    mujoco.MjrRect(0, 0, width, height),
                    self._data,
                    mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONT_NORMAL), # This needs to be properly initialized
                    img,
                    cam_id,
                )
                obs_dict[cam_name] = img
            else:
                # Return a black image if camera not found
                width = self.config.cameras[cam_name].width
                height = self.config.cameras[cam_name].height
                obs_dict[cam_name] = np.zeros((height, width, 3), dtype=np.uint8)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action = {}
        for joint_name in self.joint_names:
            key = f"{joint_name}.pos"
            if key in action:
                ctrl_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if ctrl_id != -1:
                    # Assuming direct position control for actuators
                    self._data.ctrl[ctrl_id] = action[key]
                    sent_action[key] = action[key]
                else:
                    logger.warning(f"Actuator for joint '{joint_name}' not found in MuJoCo model.")
                    sent_action[key] = 0.0 # Default value
            else:
                sent_action[key] = 0.0 # Default value if action not provided

        # Step the simulation
        mujoco.mj_step(self._model, self._data)

        return sent_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._viewer:
            self._viewer.close()
            self._viewer = None
        if self._data:
            mujoco.mj_deleteData(self._data)
            self._data = None
        self._model = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")
