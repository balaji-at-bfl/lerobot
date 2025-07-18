import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

class InteractiveRobotDemo:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and site indices
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i+1}_pos") for i in range(6)]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        
        # Control parameters
        self.target_positions = [
            [0.4, 0.0, 0.3],    # Forward
            [0.3, 0.3, 0.3],    # Forward-right
            [0.0, 0.4, 0.3],    # Right
            [-0.2, 0.3, 0.4],   # Back-right-up
            [0.2, -0.3, 0.2],   # Forward-left-down
            [0.0, 0.0, 0.5],    # Above base
        ]
        self.current_target_idx = 0
        self.moving = False
        
        # Initialize robot
        self.reset_robot()
        
    def reset_robot(self):
        """Reset robot to home position"""
        home_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.joint_ids] = home_pos
        self.data.ctrl[self.actuator_ids] = home_pos
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
            
            # Compute Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            
            # Use only the joints we control
            jac = jac_pos[:, self.joint_ids]
            
            try:
                delta_q = 0.1 * np.linalg.pinv(jac, rcond=1e-4) @ error
                new_q = self.data.qpos[self.joint_ids] + delta_q
                
                # Apply joint limits
                for j, joint_id in enumerate(self.joint_ids):
                    q_min = self.model.jnt_range[joint_id, 0]
                    q_max = self.model.jnt_range[joint_id, 1]
                    new_q[j] = np.clip(new_q[j], q_min, q_max)
                
                self.data.qpos[self.joint_ids] = new_q
                self.data.ctrl[self.actuator_ids] = new_q
                
            except np.linalg.LinAlgError:
                break
                
        return False
    
    def move_to_next_target(self):
        """Move to the next target position"""
        if self.moving:
            return
            
        self.moving = True
        target_pos = self.target_positions[self.current_target_idx]
        
        print(f"Moving to target {self.current_target_idx + 1}: {target_pos}")
        
        if self.jacobian_ik(target_pos):
            print("✓ Reached target position")
        else:
            print("✗ Failed to reach target position")
            
        # Get actual final position
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.data.site_xpos[self.ee_site_id].copy()
        error = np.linalg.norm(np.array(target_pos) - final_pos)
        print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"Position error: {error:.4f} m\n")
        
        # Move to next target
        self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
        self.moving = False
    
    def run_simulation(self):
        """Run the interactive simulation"""
        print("Starting interactive robot demo...")
        print("Controls:")
        print("- SPACE: Move to next target position")
        print("- R: Reset robot to home position")
        print("- ESC: Exit\n")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Add target visualization
            viewer.user_scn.ngeom = 0
            
            while viewer.is_running():
                # Update simulation
                mujoco.mj_step(self.model, self.data)
                
                # Add visual markers for targets
                if viewer.user_scn.ngeom < len(self.target_positions):
                    for i, pos in enumerate(self.target_positions):
                        # Add sphere marker for each target
                        if i < 6:  # MuJoCo viewer has limited user geometry slots
                            viewer.user_scn.ngeom = i + 1
                            geom = viewer.user_scn.geoms[i]
                            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                            geom.size = [0.02, 0, 0]
                            geom.pos = pos
                            if i == self.current_target_idx:
                                geom.rgba = [1, 0, 0, 0.8]  # Red for current target
                            else:
                                geom.rgba = [0, 1, 0, 0.5]  # Green for other targets
                
                # Handle keyboard input
                if viewer.is_running():
                    # Check for key presses (simplified - you may need to implement proper key handling)
                    pass
                
                viewer.sync()
                time.sleep(0.01)  # 100 Hz

def run_interactive_demo():
    """Run the interactive demo"""
    try:
        demo = InteractiveRobotDemo('dobot/scene.xml')  # Update path as needed
        
        # Run automatic demo
        print("Running automatic demo...")
        for i in range(3):  # Demo 3 movements
            time.sleep(2)  # Wait 2 seconds
            demo.move_to_next_target()
            time.sleep(1)  # Wait 1 second after movement
        
        print("\nStarting interactive mode...")
        demo.run_simulation()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your XML files are in the correct path and MuJoCo is properly installed.")

if __name__ == "__main__":
    run_interactive_demo()