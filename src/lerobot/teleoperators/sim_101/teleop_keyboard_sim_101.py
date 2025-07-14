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
from lerobot.teleoperators.keyboard import KeyboardTeleop
from .config_sim_101 import Sim101TeleopConfig

logger = logging.getLogger(__name__)


class Sim101KeyboardTeleop(KeyboardTeleop):
    """
    A teleoperator for the simulated SO-101 arm that uses the keyboard for control.
    """

    config_class = Sim101TeleopConfig
    name = "sim_101_keyboard"

    def __init__(self, config: Sim101TeleopConfig):
        super().__init__(config)
        self.joint_controller = RobotJointController()
        self.joint_positions = {name: 0.0 for name in self.joint_controller.joint_names}

        self.key_to_joint_map = {
            "q": ("shoulder_pan", 1),
            "a": ("shoulder_pan", -1),
            "w": ("shoulder_lift", 1),
            "s": ("shoulder_lift", -1),
            "e": ("elbow_flex", 1),
            "d": ("elbow_flex", -1),
            "r": ("wrist_flex", 1),
            "f": ("wrist_flex", -1),
            "t": ("wrist_roll", 1),
            "g": ("wrist_roll", -1),
            "y": ("gripper", 1),
            "h": ("gripper", -1),
        }

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{motor}.pos": float for motor in self.joint_controller.joint_names}

    def get_action(self) -> dict[str, Any]:
        self._drain_pressed_keys()

        for key, (joint_name, direction) in self.key_to_joint_map.items():
            if self.current_pressed.get(key):
                self.joint_positions[joint_name] += direction * 0.5  # Increment/decrement by a small amount

        # Clamp the joint positions to the normalized range
        for joint_name, pos in self.joint_positions.items():
            self.joint_positions[joint_name] = np.clip(
                pos, self.joint_controller.target_min, self.joint_controller.target_max
            )

        return {f"{name}.pos": val for name, val in self.joint_positions.items()}
