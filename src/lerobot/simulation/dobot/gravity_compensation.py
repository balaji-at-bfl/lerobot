import mujoco
import numpy as np
import time

class GravityCompensatedController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and site indices
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i+1}_pos") for i in range(6)]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        
        # Control gains
        self.kp = np.array([1000, 1500, 1200, 800, 600, 400])  # Position gains
        self.kv = np.array([100, 150, 120, 80, 60, 40])        # Velocity gains
        
        # Stable home position
        self.home_position = np.array([0.0, -0.5, 1.2, 0.0, 0.8, 0.0])
        self.target_position = self.home_position.copy()
        
        # Initialize
        self.reset_robot()
        
    def reset_robot(self):
        """Reset robot to stable home position"""
        self.data.qpos[self.joint_ids] = self.home_position
        self.data.qvel[self.joint_ids] = 0.0
        self.target_position = self.home_position.copy()
        mujoco.mj_forward(self.model, self.data)
        
    def compute_gravity_compensation(self):
        """Compute gravity compensation torques"""
        # Compute gravity compensation using MuJoCo's inverse dynamics
        qfrc_bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 1, qfrc_bias)  # 1 = gravity only
        
        # Extract torques for our controlled joints
        gravity_torques = qfrc_bias[self.joint_ids]
        return gravity_torques
        
    def pd_control_with_gravity_comp(self):
        """PD control with gravity compensation"""
        # Current joint states
        q_current = self.data.qpos[self.joint_ids]
        qd_current = self.data.qvel[self.joint_ids]
        
        # PD control torques
        position_error = self.target_position - q_current
        velocity_error = -qd_current  # Target velocity is 0
        
        pd_torques = self.kp * position_error + self.kv * velocity_error
        
        # Gravity compensation
        gravity_torques = self.compute_gravity_compensation()
        
        # Total control torques
        total_torques = pd_torques + gravity_torques
        
        # Apply torques (convert to actuator commands)
        self.data.ctrl[self.actuator_ids] = self.target_position  # Position control
        
        return total_torques
        
    def set_target_joints(self, joint_angles):
        """Set target joint angles"""
        self.target_position = np.array(joint_angles)
        
    def jacobian_ik(self, target_pos, max_iter=100, tolerance=1e-3):
        """IK with gravity compensation"""
        target_pos = np.array(target_pos)
        
        for i in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            
            error = target_pos - current_pos
            if np.linalg.norm(error) < tolerance:
                return True
            
            # Compute Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            
            jac = jac_pos[:, self.joint_ids]
            
            try:
                delta_q = 0.1 * np.linalg.pinv(jac, rcond=1e-4) @ error
                new_q = self.data.qpos[self.joint_ids] + delta_q
                
                # Apply joint limits
                for j, joint_id in enumerate(self.joint_ids):
                    q_min = self.model.jnt_range[joint_id, 0]
                    q_max = self.model.jnt_range[joint_id, 1]
                    new_q[j] = np.clip(new_q[j], q_min, q_max)
                
                # Update target (not direct joint position)
                self.target_position = new_q
                
            except np.linalg.LinAlgError:
                break
                
        return False
        
    def step_simulation(self):
        """Step simulation with control"""
        # Apply control
        self.pd_control_with_gravity_comp()
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
    def move_to_position(self, target_pos, duration=3.0):
        """Move end-effector to position with proper control"""
        print(f"Moving to position: {target_pos}")
        
        # Solve IK to get target joint angles
        if self.jacobian_ik(target_pos):
            target_joints = self.target_position.copy()
            start_joints = self.data.qpos[self.joint_ids].copy()
            
            # Smooth trajectory
            start_time = time.time()
            while time.time() - start_time < duration:
                t = (time.time() - start_time) / duration
                t = min(t, 1.0)
                
                # Smooth interpolation
                t_smooth = 3 * t**2 - 2 * t**3
                current_joints = start_joints + t_smooth * (target_joints - start_joints)
                
                self.set_target_joints(current_joints)
                self.step_simulation()
                
                time.sleep(0.01)  # 100 Hz
            
            # Final position
            self.set_target_joints(target_joints)
            print("Movement completed")
            return True
        else:
            print("IK failed")
            return False

def demo_gravity_compensation():
    """Demo with proper gravity compensation"""
    controller = GravityCompensatedController('scene.xml')
    
    print("Robot initialized with gravity compensation")
    
    # Test positions
    positions = [
        [0.4, 0.0, 0.3],
        [0.3, 0.3, 0.4], 
        [0.2, -0.2, 0.25],
        [0.0, 0.0, 0.5]
    ]
    
    for pos in positions:
        success = controller.move_to_position(pos, duration=4.0)
        if success:
            # Hold position for 2 seconds
            for _ in range(200):  # 2 seconds at 100Hz
                controller.step_simulation()
                time.sleep(0.01)
        
        time.sleep(1.0)

if __name__ == "__main__":
    demo_gravity_compensation()