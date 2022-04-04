from locale import normalize
import numpy as np
import robosuite as suite
import os
import yaml

import wandb
from wandb.integration.sb3 import WandbCallback

from robosuite.models.robots.robot_model import register_robot
from robosuite.environments.base import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from src.environments import Lift_4_objects, Lift_edit
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.helper_functions.wrap_env import make_multiprocess_env, make_singel_env
from src.helper_functions.camera_functions import adjust_width_of_image


import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
def display_video(frames, framerate=2):
    print(frames.shape)
    print(frames[0].shape)
    height, width, __ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    writer = animation.writers['ffmpeg'](fps=30)
    anim.save(video_name,writer=writer,dpi=dpi)

from scipy import ndimage



if __name__ == '__main__':
    register_robot(IIWA_14)
    register_gripper(Robotiq85Gripper_iiwa_14)
    register_robot_class_mapping("IIWA_14")
    register_env(Lift_edit)
    register_env(Lift_4_objects)

    with open("rl_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Environment specifications
    env_options = config["robosuite"]
    env_options["camera_widths"] = adjust_width_of_image(env_options["camera_heights"])
    env_options["custom_camera_trans_matrix"] = np.array(env_options["custom_camera_trans_matrix"])
    env_id = env_options.pop("env_id")

    # Observations
    obs_config = config["gymwrapper"]
    obs_list = obs_config["observations"] 
    smaller_action_space = obs_config["smaller_action_space"]

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    check_pt_interval = sb_config["check_pt_interval"]
    num_procs = sb_config["num_procs"]

    messages_to_wand_callback = config["wandb_callback"]
    messages_to_eval_callback = config["eval_callback"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    tb_log_folder = file_handling["tb_log_folder"]
    tb_log_name = file_handling["tb_log_name"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]

    continue_training_model_folder = file_handling["continue_training_model_folder"]
    continue_training_model_filename = file_handling["continue_training_model_filename"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]


    print("making")
    env = make_singel_env(env_id, env_options, obs_list, smaller_action_space)


    env = VecTransposeImage(env)

    # Create model
    model = PPO(policy_type, env= env, **policy_kwargs)
    obs = env.reset()
    action, _states = model.predict(obs)
    print(f"action: {action}")
    obs, reward, done, info = env.step(action)
    print("Created a new model")

    #print(env.observation_space)
    #print(obs['custom_image'].shape)

    print(obs['custom_image'])
    



    ### Video
    num_episodes = 1
    video_name = 'test.mp4'
    frames = []
    video = False
    if video == True:
        for i in range(num_episodes):
            obs = env.reset()
            img = obs['custom_image']
            rotated_img = ndimage.rotate(img, 180)
            
            frames.append(rotated_img)
            for i in range(5):
                chosen_action = model.predict(obs)
                obs,reward,done,info = env.step(chosen_action[0])
                img = obs['custom_image']
                rotated_img = ndimage.rotate(img, 180)
                frames.append(rotated_img)
        display_video(frames)

    env.close()

