import time
from lerobot.robots.dobot import Dobot, DobotConfig
from lerobot.teleoperators.dobot import DobotTeleop, DobotTeleopConfig

def main():
    # Initialize robot and teleoperator
    robot_config = DobotConfig()
    robot = Dobot(robot_config)

    teleop_config = DobotTeleopConfig()
    teleop = DobotTeleop(teleop_config)
    teleop.set_robot(robot)

    print("Starting teleoperation...")
    teleop.start_teleop()

    try:
        while teleop.control_on:
            action = teleop.get_action()
            if action is not None:
                robot.apply_action(action)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping teleoperation.")
    finally:
        robot.close()
        print("Robot connection closed.")

if __name__ == "__main__":
    main()
