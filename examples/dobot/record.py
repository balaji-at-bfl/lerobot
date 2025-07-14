import time
import cv2
from lerobot.robots.dobot import Dobot, DobotConfig
from lerobot.teleoperators.dobot import DobotTeleop, DobotTeleopConfig
from lerobot.cameras.opencv import CameraOpenCV
from lerobot.datasets.image_writer import ImageWriter
import os
from datetime import datetime
import csv

def main():
    # Initialize robot and teleoperator
    robot_config = DobotConfig()
    robot = Dobot(robot_config)

    teleop_config = DobotTeleopConfig()
    teleop = DobotTeleop(teleop_config)
    teleop.set_robot(robot)

    # Initialize cameras
    top_camera = CameraOpenCV({"name": "opencv_camera", "video_index": 0})
    wrist_camera = CameraOpenCV({"name": "opencv_camera", "video_index": 1})

    # Setup data recording
    base_path = "dobot_data"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(base_path, f"robot_log_{timestamp_str}.csv")
    os.makedirs(base_path, exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp',
        'obs_x', 'obs_y', 'obs_z', 'obs_rx', 'obs_ry', 'obs_rz',
        'obs_j1', 'obs_j2', 'obs_j3', 'obs_j4', 'obs_j5', 'obs_j6',
        'obs_gripper',
        'action_x', 'action_y', 'action_z', 'action_rx', 'action_ry', 'action_rz',
        'action_gripper',
    ])

    top_video_writer = ImageWriter(os.path.join(base_path, f"top_camera_{timestamp_str}.mp4"), top_camera.fps)
    wrist_video_writer = ImageWriter(os.path.join(base_path, f"wrist_camera_{timestamp_str}.mp4"), wrist_camera.fps)

    print("Starting data recording...")
    teleop.start_teleop()

    try:
        while teleop.control_on:
            start_time = time.time()

            # Get observation and action
            obs_pose, obs_angles = robot.get_observation()
            action = teleop.get_action()

            if action is not None:
                robot.apply_action(action)

                # Record data
                top_frame = top_camera.read()
                wrist_frame = wrist_camera.read()

                if top_frame is not None and wrist_frame is not None:
                    timestamp = time.time()
                    csv_writer.writerow([
                        f"{timestamp:.4f}",
                        *obs_pose,
                        *obs_angles,
                        teleop.gripper_on,
                        *action,
                        teleop.gripper_on,
                    ])
                    top_video_writer.write(top_frame)
                    wrist_video_writer.write(wrist_frame)

            # Maintain recording frequency
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / 15.0) - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping data recording.")
    finally:
        robot.close()
        top_camera.close()
        wrist_camera.close()
        top_video_writer.close()
        wrist_video_writer.close()
        csv_file.close()
        print("Robot and cameras closed.")

if __name__ == "__main__":
    main()
