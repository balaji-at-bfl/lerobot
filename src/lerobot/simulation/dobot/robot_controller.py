 import mujoco
import numpy as np
import time

class DobotController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and site indices
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i+1}_pos") for i in range(6)]
        
        # End-effector site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        
        # Initialize to a stable home position (not all zeros)
        self.home_position = np.array([0.0, -0.5, 1.2, 0.0, 0.8, 0.0])  # Stable upright pose
        self.current_target = self.home_position.copy()
        
    def reset_to_home(self):
        """Reset robot to home position"""
        self.data.qpos[self.joint_ids] = self.home_position
        self.data.ctrl[self.actuator_ids] = self.home_position
        mujoco.mj_forward(self.model, self.data)
        
    def get_end_effector_pose(self):
        """Get current end-effector position and orientation"""
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return pos, rot_mat
    
    def set_joint_positions(self, joint_positions):
        """Set target joint positions"""
        self.current_target = np.array(joint_positions)
        self.data.ctrl[self.actuator_ids] = self.current_target
        
    def get_joint_positions(self):
        """Get current joint positions"""
        return self.data.qpos[self.joint_ids].copy()
    
    def jacobian_ik(self, target_pos, max_iterations=100, tolerance=1e-3, step_size=0.1):
        """
        Simple Jacobian-based inverse kinematics
        
        Args:
            target_pos: Target position [x, y, z]
            max_iterations: Maximum IK iterations
            tolerance: Position tolerance for convergence
            step_size: IK step size
        """
        target_pos = np.array(target_pos)
        
        for i in range(max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            
            # Check convergence
            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                print(f"IK converged in {i} iterations")
                return True
            
            # Compute Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            
            # Use only position jacobian for the joints we control
            jac = jac_pos[:, self.joint_ids]
            
            # Pseudo-inverse solution
            try:
                jac_pinv = np.linalg.pinv(jac, rcond=1e-4)
                delta_q = step_size * jac_pinv @ error
                
                # Update joint positions
                new_q = self.data.qpos[self.joint_ids] + delta_q
                
                # Apply joint limits
                for j, joint_id in enumerate(self.joint_ids):
                    q_min = self.model.jnt_range[joint_id, 0]
                    q_max = self.model.jnt_range[joint_id, 1]
                    new_q[j] = np.clip(new_q[j], q_min, q_max)
                
                # Update simulation
                self.data.qpos[self.joint_ids] = new_q
                self.data.ctrl[self.actuator_ids] = new_q
                
            except np.linalg.LinAlgError:
                print("Jacobian singular, stopping IK")
                break
                
        print(f"IK did not converge after {max_iterations} iterations")
        return False
    
    def move_to_position(self, target_pos, duration=2.0):
        """
        Move end-effector to target position smoothly
        
        Args:
            target_pos: Target position [x, y, z]
            duration: Movement duration in seconds
        """
        print(f"Moving to position: {target_pos}")
        
        # Solve IK
        if self.jacobian_ik(target_pos):
            target_joints = self.data.qpos[self.joint_ids].copy()
            start_joints = self.current_target.copy()
            
            # Smooth interpolation
            start_time = time.time()
            while time.time() - start_time < duration:
                t = (time.time() - start_time) / duration
                t = min(t, 1.0)
                
                # Smooth interpolation (ease-in-out)
                t_smooth = 3 * t**2 - 2 * t**3
                
                current_joints = start_joints + t_smooth * (target_joints - start_joints)
                self.set_joint_positions(current_joints)
                
                time.sleep(0.02)  # 50 Hz control rate
            
            # Ensure final position
            self.set_joint_positions(target_joints)
            print("Movement completed")
        else:
            print("Failed to find IK solution")
    
    def step_simulation(self):
        """Step the simulation forward"""
        mujoco.mj_step(self.model, self.data)
        
        
def run_demo():
    """Demo script showing robot control"""
    controller = DobotController('scene.xml')  # Update path as needed
    
    # Reset to home
    controller.reset_to_home()
    
    # Example usage
    print("Robot initialized. Current end-effector position:")
    pos, _ = controller.get_end_effector_pose()
    print(f"Position: {pos}")
    
    # Move to different positions
    target_positions = [
        [0.4, 0.2, 0.3],   # Forward and right
        [0.3, -0.2, 0.4],  # Forward and left, higher
        [0.2, 0.0, 0.2],   # Close to base
        [0.0, 0.0, 0.5],   # Directly above base
    ]
    
    for target_pos in target_positions:
        print(f"\nMoving to: {target_pos}")
        controller.move_to_position(target_pos, duration=3.0)
        
        # Verify position
        final_pos, _ = controller.get_end_effector_pose()
        error = np.linalg.norm(np.array(target_pos) - final_pos)
        print(f"Final position: {final_pos}")
        print(f"Position error: {error:.4f} m")
        
        time.sleep(1.0)  # Pause between movements

if __name__ == "__main__":
    run_demo()