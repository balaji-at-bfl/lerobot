import time
import numpy as np
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from inputs import get_gamepad
import threading

class DobotTeleop(GamepadTeleop):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = None 
        self.control_on = True
        self.last_pose = [0,0,0,0,0,0]
        self.movement_scale = {'x': 0.5, 'y': 0.5, 'z': 0.5, 'rx': 1, 'ry': 1, 'rz': 1}
        self.gripper_on = False
        self.axis_map = {'ABS_X': 'y', 'ABS_Y': 'x', 'ABS_Z': 'rz', 'ABS_RX': 'ry', 'ABS_RY': 'z', 'ABS_RZ': 'rx'}
        self.button_map = {'BTN_TR': self.toggle_gripper, 'BTN_START': self.stop_control}
        self.axis_states = {ax: 0 for ax in self.axis_map.keys()}

    def set_robot(self, robot):
        self.robot = robot
        self.last_pose, _ = self.robot.get_observation()

    def toggle_gripper(self):
        self.gripper_on = not self.gripper_on
        self.robot.toggle_gripper()
        print(f"Gripper {'on' if self.gripper_on else 'off'}")

    def stop_control(self):
        self.control_on = False
        print("Stopping teleoperation.")

    def _gamepad_reader(self):
        while self.control_on:
            try:
                events = get_gamepad()
                for event in events:
                    if event.ev_type == 'Absolute':
                        self.axis_states[event.code] = event.state
                    elif event.ev_type == 'Key' and event.state == 1:
                        if event.code in self.button_map:
                            self.button_map[event.code]()
            except Exception as e:
                print(f"Error reading gamepad: {e}")
                self.control_on = False

    def start_teleop(self):
        threading.Thread(target=self._gamepad_reader, daemon=True).start()
        print("Gamepad teleoperation started.")

    def get_action(self):
        if not self.control_on:
            return None

        delta = {axis: 0 for axis in ['x', 'y', 'z', 'rx', 'ry', 'rz']}

        for code, axis_name in self.axis_map.items():
            val = self.axis_states[code]
            # Normalize and apply deadzone
            norm_val = val / 32768.0
            if abs(norm_val) > 0.1:
                delta[axis_name] = norm_val * self.movement_scale[axis_name]

        new_pose = self.last_pose.copy()
        new_pose[0] += delta['x']
        new_pose[1] += delta['y']
        new_pose[2] += delta['z']
        new_pose[3] += delta['rx']
        new_pose[4] += delta['ry']
        new_pose[5] += delta['rz']
        
        self.last_pose = new_pose
        return np.array(new_pose)
