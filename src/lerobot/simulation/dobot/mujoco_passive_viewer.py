import mujoco
import mujoco.viewer
import time
import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir(os.getcwd()))
# Load model and create data
model = mujoco.MjModel.from_xml_path('dobot/scene.xml')
data = mujoco.MjData(model)

# Set stable initial position
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}") for i in range(6)]
stable_pos = [0.0, -0.5, 1.2, 0.0, 0.8, 0.0]
data.qpos[joint_ids] = stable_pos
data.ctrl[:6] = stable_pos  # Set control targets

# Launch passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset the simulation to apply initial conditions
    mujoco.mj_resetData(model, data)
    data.qpos[joint_ids] = stable_pos
    data.ctrl[:6] = stable_pos
    
    print("MuJoCo viewer launched. Robot should hold position without drooping!")
    print("Close the viewer window to exit.")
    
    # Run simulation with viewer updates
    start_time = time.time()
    while viewer.is_running():
        step_start = time.time()
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer (viewer will automatically render at ~60fps)
        viewer.sync()
        
        # Control loop timing (maintain real-time simulation)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation ended.")