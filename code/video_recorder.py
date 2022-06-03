from email import policy
from locale import normalize
from unicodedata import name
import numpy as np
import robosuite as suite
import os
import yaml

import wandb
from wandb.integration.sb3 import WandbCallback

from robosuite.models.robots.robot_model import register_robot
from robosuite.environments.base import register_env

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from src.callback.progresscallback import CustomEvalCallback
from src.environments import Lift_4_objects, Lift_edit, Lift_edit_green, Lift_edit_multiple_objects
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14, IIWA_14_modified, IIWA_14_modified_flange
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14, Robotiq85Gripper_iiwa_14_longer_finger
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.helper_functions.wrap_env import make_multiprocess_env, make_env
from src.helper_functions.camera_functions import adjust_width_of_image
from src.helper_functions.hyperparameters import linear_schedule_1, linear_schedule_2
from src.helper_functions.customCombinedExtractor import CustomCombinedExtractor, LargeCombinedExtractor, CustomCombinedExtractor_object_obs
from src.helper_functions.customCombinedSurreal import CustomCombinedSurreal

import numpy as np
import imageio
import robosuite.utils.macros as macros
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image

def record_video(env, model, video_length,num_episodes, fps, name_of_video_file):
    macros.IMAGE_CONVENTION = "opencv"
    # create a video writer with imageio
    os.mkdir(name_of_video_file)
    writer = imageio.get_writer(name_of_video_file + "/video.mp4", fps=fps)

    for j in range(num_episodes):
        obs = env.reset()
        reward_plot = []
        step_plot = []
        for i in range(video_length+10):

            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_plot.append(env.get_original_reward())
            step_plot.append(i)
            frame = obs["custom_image_rgbd"][0,:,:,:3]
            img = Image.fromarray(frame, 'RGB')
            img = img.rotate(180)

            frame = np.asarray(img)
            #frame = ndimage.rotate(frame, 180)
            writer.append_data(frame)
            if done:
                plt.plot(step_plot,reward_plot)
                plt.ylabel('Reward')
                plt.xlabel('Timestep')
                plt.title(str(j) + "epsiode")
                plt.savefig(name_of_video_file + "/plot_" + str(j+1))
                plt.clf()
                break

    writer.close()


if __name__ == '__main__':
    register_robot(IIWA_14_modified)
    register_robot(IIWA_14)
    register_robot(IIWA_14_modified_flange)
    register_gripper(Robotiq85Gripper_iiwa_14)
    register_gripper(Robotiq85Gripper_iiwa_14_longer_finger)
    register_robot_class_mapping("IIWA_14")
    register_robot_class_mapping("IIWA_14_modified")
    register_robot_class_mapping("IIWA_14_modified_flange")
    register_env(Lift_edit)
    register_env(Lift_4_objects)
    register_env(Lift_edit_green)
    register_env(Lift_edit_multiple_objects)

    yaml_file = "config_files/" + input("Which yaml file to load config from: ")
    #yaml_file = "config_files/sac_baseline_rgbd_uint8.yaml"
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    domain_yaml_file = "config_files/domain_rand_args.yaml"
    with open(domain_yaml_file, 'r') as stream:
        domain_config = yaml.safe_load(stream)

    answer = input("Have you dobbel checked if you are using the correct load and save files? \n  [y/n] ") 
    if answer != "y":
         exit()


    # Environment specifications
    env_options = config["robosuite"]
    if env_options["custom_camera_conversion"]:
        env_options["camera_widths"] = adjust_width_of_image(env_options["camera_heights"])
    env_options["custom_camera_trans_matrix"] = np.array(env_options["custom_camera_trans_matrix"])
    env_id = env_options.pop("env_id")

    #normalize obs and rew
    normalize_obs = config['normalize_obs']
    normalize_rew = config['normalize_rew']
    norm_obs_keys = config['norm_obs_keys']
    
    #use domain randomization
    use_domain_rand = config["use_domain_rand"]
    domain_rand_args = domain_config["domain_rand_args"]

    if use_domain_rand:
        print("using domain randomization")
    # Observations
    obs_config = config["gymwrapper"]
    obs_list = obs_config["observations"] 
    smaller_action_space = obs_config["smaller_action_space"]
    xyz_action_space = obs_config["xyz_action_space"]
    use_rgbd = obs_config['rgbd']
    close_img = obs_config['close_img']

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    num_procs = sb_config["num_procs"]
    policy = sb_config['policy']



    messages_to_wand_callback = config["wandb_callback"]
    messages_to_eval_callback = config["eval_callback"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    #Implementing custom feature extractor
    if policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'large':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = LargeCombinedExtractor
    elif policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'small':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = CustomCombinedExtractor
    else: policy_kwargs["policy_kwargs"].pop("features_extractor_class")

    print(policy_kwargs["policy_kwargs"])

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]
    best_model= config["eval_callback"]
    best_model_save_path = best_model['best_model_save_path']

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]

    control_freq = env_options['control_freq']
    horizon = env_options['horizon']


    #Create ENV
    print("making")
    
    num_procs = 1
    #env = VecTransposeImage(SubprocVecEnv([make_multiprocess_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space,  i, seed, use_domain_rand=use_domain_rand, domain_rand_args=domain_rand_args) for i in range(num_procs)]))
    env = make_multiprocess_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, seed, use_domain_rand, domain_rand_args,num_procs)
    env = SubprocVecEnv(env)


    load_model_path = os.path.join(best_model_save_path, 'best_model.zip')
    load_vecnormalize_path = os.path.join(best_model_save_path, 'vec_normalize_best_model.pkl')


    if (normalize_obs or normalize_rew):
        env = VecNormalize.load(load_vecnormalize_path, env)
    else: 
        exit("Write either make_new or load_file")

    
    # Load model
    if policy == 'PPO':
            model = PPO.load(load_model_path, env=env)
    elif policy == 'SAC':
        model = SAC.load(load_model_path, env=env)
        
    env.training = False

    num_episodes = int(input("How many epsiodes do you want to record?   "))
    name_of_video_file = input("What should video file be called?   ") 
    


    record_video(
        env=env, 
        model=model,
        video_length = horizon, 
        num_episodes= num_episodes, 
        name_of_video_file=name_of_video_file,
        fps = control_freq)
    env.close()