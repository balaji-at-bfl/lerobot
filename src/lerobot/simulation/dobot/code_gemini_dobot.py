import mujoco
import mujoco.viewer
import numpy as np
import time

class RobotController:  # Or rename it to DobotController if you prefer
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i+1}_pos") for i in range(6)]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        self.home_position = np.array([0.0, -0.5, 1.2, 0.0, 0.8, 0.0])
        self.reset_to_home()

    def reset_to_home(self):
        """Reset robot to home position"""
        self.data.qpos[self.joint_ids] = self.home_position
        self.data.ctrl[self.actuator_ids] = self.home_position
        mujoco.mj_forward(self.model, self.data)

    def jacobian_ik(self, target_pos, max_iter=50, tolerance=1e-3):
        """Simplified IK solver"""
        target_pos = np.array(target_pos)
        for i in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.ee_site_id].copy()

            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                return True

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            jac = jac_pos[:, self.joint_ids]

            try:
                delta_q = 0.1 * np.linalg.pinv(jac, rcond=1e-4) @ error
                new_q = self.data.qpos[self.joint_ids] + delta_q

                for j, joint_id in enumerate(self.joint_ids):
                    q_min = self.model.jnt_range[joint_id, 0]
                    q_max = self.model.jnt_range[joint_id, 1]
                    new_q[j] = np.clip(new_q[j], q_min, q_max)

                self.data.qpos[self.joint_ids] = new_q
                self.data.ctrl[self.actuator_ids] = new_q

            except np.linalg.LinAlgError:
                break

        return False


    def move_to(self, x, y, z):
        """Move the end-effector to the specified Cartesian position."""
        target_pos = np.array([x, y, z])
        success = self.jacobian_ik(target_pos)

        if success:
            print(f"Moved to position: {target_pos}")
        else:
            print(f"IK failed for position: {target_pos}")

    def step_simulation(self):
        """Step the simulation forward."""
        mujoco.mj_step(self.model, self.data)

def main():
    # Load your MuJoCo model
    model_path = 'dobot/scene.xml'  # Or 'dobot/scene.xml' if 'dobot' is a subdirectory
    controller = RobotController(model_path)

    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        viewer.cam.distance = 2  # Adjust as needed
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20

        # Example usage:  Call move_to from the terminal (or within the script)
        target_positions = [
            [0.4, 0.0, 0.3],    # Forward
            [0.3, 0.3, 0.3],    # Forward-right
            [0.0, 0.4, 0.3],    # Right
            [-0.2, 0.3, 0.4],   # Back-right-up
            [0.2, -0.3, 0.2],   # Forward-left-down
            [0.0, 0.0, 0.5],    # Above base
        ]

        for target in target_positions:
            controller.move_to(target[0], target[1], target[2])

            # Simulate for a few steps to let the robot settle
            for _ in range(50):
                controller.step_simulation()
                time.sleep(0.02)  # Control rate (e.g., 50 Hz)
        
            # Keep the simulation running
            while viewer.is_running():
                controller.step_simulation()
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    main()