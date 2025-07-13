from loop_rate_limiters import RateLimiter
import numpy as np
import torch
import logging

import mujoco
import mujoco.viewer
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.simulation.norm_denorm import RobotJointController
import time

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inference(task="idle"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    controller = RobotJointController()
    try:
        model = mujoco.MjModel.from_xml_path("/home/bfl3/bfl_works/new_lerobot/lerobot/src/lerobot/simulation/SO101/scene.xml")
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 480, 640)
    except Exception as e:
        logger.critical(f"Failed to load MuJoCo model or create renderer: {e}")
        return

    logger.info("About to load the pretrained model.")
    try:
        policy = SmolVLAPolicy.from_pretrained("/home/bfl3/bfl_works/new_lerobot/lerobot/models/train/my_smolvla1/checkpoints/last/pretrained_model")
        logger.info(f"loaded and moving to {device}")
        policy.to(device)
        policy.eval()
        logger.info(f"Pretrained model loaded successfully and moved to {device}.")
    except Exception as e:
        logger.critical(f"Failed to load policy: {e}")
        return

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    rate = RateLimiter(30)

    viewer = None
    try:
        with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            while viewer.is_running():
                observation_np = {}
                observation_np["agent_pos"] = controller._normalize_array(np.array([data.qpos[i] for i in range(6)]))

                renderer.update_scene(data, camera="camera_front")
                camera_front_img = renderer.render()
                renderer.update_scene(data, camera="camera_gripper")
                camera_gripper_img = renderer.render()

                observation_np["pixels"] = {
                    "front": camera_front_img,
                    "left": camera_gripper_img # TODO: save the curren 
                }
                observation_np["task"] = task

                # `preprocess_observation` converts numpy arrays to torch tensors on the CPU
                # and renames 'agent_pos' to 'state'.
                observation_torch = preprocess_observation(observation_np)
                
                # ------ CORRECTED AND IMPROVED DATA PREPARATION LOOP ------
                # This loop now handles moving to the device AND adding the batch dimension.
                batch = {}
                for key, value in observation_torch.items():
                    if isinstance(value, torch.Tensor):
                        # The key is now 'state', not 'agent_pos'
                        logger.debug(f"Processing key '{key}'")
                        batch[key] = value.unsqueeze(0).to(device)
                    elif isinstance(value, dict):  # Handle nested dicts like 'pixels'
                        batch[key] = {}
                        for k, v in value.items():
                            if isinstance(v, torch.Tensor):
                                logger.debug(f"Processing nested key '{key}.{k}'")
                                batch[key][k] = v.unsqueeze(0).to(device)
                    else:
                        # For non-tensor data like the task string
                        batch[key] = value
                # ------ END OF CORRECTED LOOP ------
                batch["task"] = observation_np["task"]
                # Select action (no `with torch.no_grad()` needed as it's built-in)
                action_tensor = policy.select_action(batch)

                action_np = action_tensor.squeeze(0).cpu().numpy()
                action = controller._unnormalize_array(action_np)
                action = np.clip(action, controller.min_ranges_rad, controller.max_ranges_rad)
                logger.debug(f"Clipped Action: {action}")

                data.ctrl[:] = action
                mujoco.mj_step(model, data)
                viewer.sync()
                rate.sleep()

    except Exception as e:
        logger.error(f"An error occurred during the simulation loop: {e}", exc_info=True)
    finally:
        if viewer and viewer.is_running():
            viewer.close()
        logger.info("Simulation finished or was interrupted.")


if __name__ == "__main__":
    for i in range(4):
        logger.info(f"running episode {i+1}")
        inference("move to the cup.") # TODO: Model will be loaded again and again, cahnge quickly.
