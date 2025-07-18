import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import sys

class SimpleRobotController:
    def __init__(self):
        # Load the scene
        self.model = mujoco.MjModel.from_xml_path('dobot/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # Get important IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}") for i in range(6)]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i+1}_pos") for i in range(6)]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        
        # Robot state
        self.home_pos = np.array([0.0, -0.5, 1.2, 0.0, 0.8, 0.0])
        self.target_pos = self.home_pos.copy()
        self.is_moving = False
        self.running = True
        
        # Initialize
        self.reset_to_home()
        
    def reset_to_home(self):
        """Reset robot to home position"""
        self.data.qpos[self.joint_ids] = self.home_pos
        self.data.ctrl[self.actuator_ids] = self.home_pos
        self.target_pos = self.home_pos.copy()
        mujoco.mj_forward(self.model, self.data)
        
    def solve_ik(self, target_xyz):
        """Simple IK solver"""
        target_xyz = np.array(target_xyz)
        
        for i in range(50):  # Max 50 iterations
            # Update forward kinematics
            mujoco.mj_forward(self.model, self.data)
            current_xyz = self.data.site_xpos[self.ee_site_id].copy()
            
            # Check if we're close enough
            error = target_xyz - current_xyz
            if np.linalg.norm(error) < 0.001:  # 1mm tolerance
                return True
                
            # Compute Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
            
            # Get jacobian for our 6 joints only
            J = jac_pos[:, self.joint_ids]
            
            # Solve for joint angles change
            try:
                delta_q = 0.05 * np.linalg.pinv(J) @ error
                new_q = self.data.qpos[self.joint_ids] + delta_q
                
                # Apply joint limits
                for j, joint_id in enumerate(self.joint_ids):
                    q_min = self.model.jnt_range[joint_id, 0]
                    q_max = self.model.jnt_range[joint_id, 1]
                    new_q[j] = np.clip(new_q[j], q_min, q_max)
                
                # Update robot configuration
                self.data.qpos[self.joint_ids] = new_q
                self.target_pos = new_q.copy()
                
            except:
                print("IK solver failed")
                return False
                
        return False
    
    def move_to_xyz(self, x, y, z, duration=3.0):
        """Move robot to XYZ position"""
        if self.is_moving:
            print("Robot is already moving!")
            return
            
        print(f"Moving to position: [{x:.3f}, {y:.3f}, {z:.3f}]")
        self.is_moving = True
        
        # Store starting position
        start_joints = self.target_pos.copy()
        
        # Solve IK for target position
        if self.solve_ik([x, y, z]):
            target_joints = self.target_pos.copy()
            
            # Smooth movement over duration
            start_time = time.time()
            while time.time() - start_time < duration:
                t = (time.time() - start_time) / duration
                t = min(t, 1.0)
                
                # Smooth interpolation
                t_smooth = 3*t**2 - 2*t**3  # Ease in-out
                current_joints = start_joints + t_smooth * (target_joints - start_joints)
                
                self.target_pos = current_joints
                self.data.ctrl[self.actuator_ids] = current_joints
                
                time.sleep(0.01)
            
            # Final position
            self.target_pos = target_joints
            self.data.ctrl[self.actuator_ids] = target_joints
            
            # Check final position
            mujoco.mj_forward(self.model, self.data)
            final_xyz = self.data.site_xpos[self.ee_site_id].copy()
            error = np.linalg.norm([x, y, z] - final_xyz)
            
            print(f"✓ Reached: [{final_xyz[0]:.3f}, {final_xyz[1]:.3f}, {final_xyz[2]:.3f}]")
            print(f"Error: {error:.4f}m\n")
            
        else:
            print("✗ Could not reach target position\n")
            
        self.is_moving = False
    
    def get_current_xyz(self):
        """Get current end-effector position"""
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def run_viewer(self):
        """Run MuJoCo viewer"""
        print("Starting MuJoCo viewer...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            mujoco.mj_resetData(self.model, self.data)
            self.reset_to_home()
            
            while viewer.is_running() and self.running:
                # Update control
                self.data.ctrl[self.actuator_ids] = self.target_pos
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                time.sleep(0.002)  # Match timestep
    
    def run_commands(self):
        """Handle command line input"""
        print("\n=== Simple Robot Controller ===")
        print("Commands:")
        print("  x y z       - Move to position (example: 0.4 0.2 0.3)")
        print("  home        - Go to home position")
        print("  pos         - Show current position")
        print("  quit        - Exit")
        print("\nEnter commands below:\n")
        
        while self.running:
            try:
                cmd = input(">>> ").strip().lower()
                
                if cmd in ['quit', 'q', 'exit']:
                    print("Goodbye!")
                    self.running = False
                    break
                    
                elif cmd == 'home':
                    print("Going home...")
                    self.move_to_xyz(0.0, 0.0, 0.5, duration=2.0)
                    
                elif cmd in ['pos', 'position']:
                    xyz = self.get_current_xyz()
                    print(f"Current position: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")
                    
                else:
                    # Try to parse x y z coordinates
                    parts = cmd.split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            
                            # Safety checks
                            dist = np.sqrt(x**2 + y**2 + z**2)
                            if dist > 0.7:
                                print(f"Warning: Target far from base ({dist:.2f}m)")
                            if z < 0.05:
                                print("Warning: Very low Z position")
                                
                            self.move_to_xyz(x, y, z)
                            
                        except ValueError:
                            print("Error: Invalid numbers")
                    elif cmd:
                        print("Usage: x y z (example: 0.3 0.2 0.4)")
                        
            except KeyboardInterrupt:
                print("\nExiting...")
                self.running = False
                break

def main():
    print("Initializing robot controller...")
    
    try:
        robot = SimpleRobotController()
        print("✓ Robot initialized successfully!")
        
        # Start viewer in background thread
        viewer_thread = threading.Thread(target=robot.run_viewer, daemon=True)
        viewer_thread.start()
        
        # Wait for viewer to start
        time.sleep(2)
        
        # Run command interface
        robot.run_commands()
        
    except FileNotFoundError:
        print("Error: Could not find 'scene.xml'")
        print("Make sure the file is in the current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()