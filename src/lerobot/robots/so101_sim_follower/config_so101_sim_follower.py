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

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig


@dataclass
class SO101SimFollowerConfig(SO101FollowerConfig):
    """
    Configuration class for the simulated SO-101 Follower Arm.
    Inherits from SO101FollowerConfig to reuse common parameters.
    """

    name: str = "so101_sim_follower"
    # Path to the MuJoCo XML model file for the SO101 arm.
    mujoco_model_path: Path = Path("simulations/SO101/so101_new_calib.xml")
