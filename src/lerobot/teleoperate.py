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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true \
    --robot.type=so101_follower \
    --teleop.type=so101_leader
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    so101_sim_follower, # New import
)
from lerobot.robots.so101_sim_follower.so101_sim_follower import SO101SimFollower # New import
from lerobot.robots.so101_follower.so101_follower import SO101Follower # New import
from lerobot.robots.so101_sim_follower.config_so101_sim_follower import SO101SimFollowerConfig # New import
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig # New import
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Enable simulation mode.
    enable_sim: bool = False
    # Disable real follower and use simulated follower instead (only with --enable-sim).
    disable_follower: bool = False

    def __post_init__(self):
        if self.disable_follower and not self.enable_sim:
            raise ValueError("--disable-follower can only be used with --enable-sim=true")

        if self.robot is None:
            raise ValueError("robot must be specified")

        if self.teleop is None:
            raise ValueError("teleop must be specified")


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot, # This is robot_for_obs
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    robot_to_control: Robot | None = None,
    sim_robot: SO101SimFollower | None = None,
    real_robot: SO101Follower | None = None,
    enable_sim: bool = False,
    disable_follower: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
            log_rerun_data(observation, action)

        robot_to_control.send_action(action)
        # If simulation is enabled and real follower is also active, mirror actions to sim robot
        if enable_sim and not disable_follower and sim_robot and real_robot:
            sim_robot.send_action(action) # Send the same action to the simulated robot
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    real_robot = None
    sim_robot = None

    if cfg.enable_sim:
        sim_robot_config = SO101SimFollowerConfig(
            id=cfg.robot.id,
            cameras=cfg.robot.cameras,
            port=cfg.robot.port,
            use_degrees=cfg.robot.use_degrees,
            max_relative_target=cfg.robot.max_relative_target,
            disable_torque_on_disconnect=cfg.robot.disable_torque_on_disconnect,
            calibration_dir=cfg.robot.calibration_dir,
        )
        sim_robot = make_robot_from_config(sim_robot_config)

        if not cfg.disable_follower:
            real_robot = make_robot_from_config(cfg.robot)
            robot_to_control = real_robot
            robot_for_obs = real_robot
        else:
            robot_to_control = sim_robot
            robot_for_obs = sim_robot
    else:
        real_robot = make_robot_from_config(cfg.robot)
        robot_to_control = real_robot
        robot_for_obs = real_robot

    teleop = make_teleoperator_from_config(cfg.teleop)

    if real_robot:
        real_robot.connect()
    if sim_robot:
        sim_robot.connect()
    teleop.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot_for_obs, # Observations from this robot
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            robot_to_control=robot_to_control, # Actions sent to this robot
            sim_robot=sim_robot, # For mirroring actions
            real_robot=real_robot, # For mirroring actions
            enable_sim=cfg.enable_sim,
            disable_follower=cfg.disable_follower,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        if real_robot:
            real_robot.disconnect()
        if sim_robot:
            sim_robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    teleoperate()
