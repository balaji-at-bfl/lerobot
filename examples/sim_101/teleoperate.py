import time

from lerobot.robots.sim_101 import Sim101, Sim101Config
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = Sim101Config(id="sim_101_robot")
teleop_config = SO101LeaderConfig(id="so101_leader_arm", port="/dev/ttyACM0")

robot = Sim101(robot_config)
teleop = SO101Leader(teleop_config)

robot.connect()
teleop.connect()

_init_rerun(session_name="sim_101_teleop")

if not robot.is_connected or not teleop.is_connected:
    raise ValueError("Robot or teleoperator is not connected!")

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()
    action = teleop.get_action()

    log_rerun_data(observation, action)

    robot.send_action(action)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
