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
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.simulation.norm_denorm import RobotJointController
from ..teleoperator import Teleoperator
from .config_sim_101 import Sim101TeleopConfig

logger = logging.getLogger(__name__)


class Sim101Teleop(Teleoperator):
    """
    A teleoperator for the simulated SO-101 arm that generates random actions.
    """

    config_class = Sim101TeleopConfig
    name = "sim_101"

    def __init__(self, config: Sim101TeleopConfig):
        super().__init__(config)
        self.joint_controller = RobotJointController()

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{motor}.pos": float for motor in self.joint_controller.joint_names}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        # The simulation teleoperator is always "connected"
        return True

    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(f"Calibration is not required for {self.name}.")

    def configure(self) -> None:
        logger.info(f"Configuration is not required for {self.name}.")

    def get_action(self) -> dict[str, Any]:
        """
        Generate a random action for the simulated robot.
        """
        # Generate random joint positions within the normalized range [-100, 100]
        random_normalized_angles = np.random.uniform(
            self.joint_controller.target_min,
            self.joint_controller.target_max,
            len(self.joint_controller.joint_names),
        )

        action = {
            f"{name}.pos": val
            for name, val in zip(self.joint_controller.joint_names, random_normalized_angles)
        }
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        logger.info(f"{self} disconnected.")
