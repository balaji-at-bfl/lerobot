import mujoco
import mujoco.viewer   # nice GPU viewer
import numpy as np




# Example target pose for the six position servos (rad)
def setup_and_move(targets, path=''):
    model = mujoco.MjModel.from_xml_path(r"B:\SO-ARM100\Simulation\SO101\so101_new_calib.xml")
    data = mujoco.MjData(model)
    data.ctrl[:] = targets                   # ← THE CONTROL “COMMAND”
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(1_000):
            mujoco.mj_step(model, data)      # advances the simulation
            viewer.sync()