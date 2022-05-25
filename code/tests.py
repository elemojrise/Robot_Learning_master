from email import policy
from locale import normalize
from tokenize import PlainToken
from unicodedata import name
import numpy as np
import robosuite as suite
import os
import yaml

from PIL import Image

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
from src.environments import Lift_4_objects, Lift_edit, Lift_edit_green
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14, IIWA_14_modified, IIWA_14_modified_flange
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14, Robotiq85Gripper_iiwa_14_longer_finger
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.helper_functions.wrap_env import make_multiprocess_env, make_env
from src.helper_functions.camera_functions import adjust_width_of_image
from src.helper_functions.hyperparameters import linear_schedule
from src.helper_functions.customCombinedExtractor import LargeCombinedExtractor, CustomCombinedExtractor_object_obs
from src.helper_functions.customCombinedSurreal import CustomCombinedSurreal



#temp
from src.wrapper.GymWrapper_multiinput_RGBD import GymWrapper_multiinput_RGBD
from src.wrapper.GymWrapper_multiinput import GymWrapper_multiinput
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

    yaml_file = "config_files/" + input("Which yaml file to load config from: ")
    #yaml_file = "config_files/ppo_baseline_rgbd_uint8_472.yaml"
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    domain_yaml_file = "config_files/domain_rand_args.yaml"
    with open(domain_yaml_file, 'r') as stream:
        domain_config = yaml.safe_load(stream)

    #answer = input("Have you dobbel checked if you are using the correct load and save files? \n  [y/n] ") 
    #if answer != "y":
    #    exit()


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

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    num_procs = sb_config["num_procs"]
    policy = sb_config['policy']

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    #Eval callback
    messages_to_eval_callback = config["eval_callback"]


    #Implementing learning rate schedular if 
    if config["learning_rate_schedular"]:
        policy_kwargs["learning_rate"] = linear_schedule(policy_kwargs["learning_rate"])
    
    #Implementing custom feature extractor
    if policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'large':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = LargeCombinedExtractor
    #elif policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'small':
    #    policy_kwargs["policy_kwargs"]["features_extractor_class"] = CustomCombinedExtractor
    else: policy_kwargs["policy_kwargs"].pop("features_extractor_class")

    print(policy_kwargs["policy_kwargs"])

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]

    # RL pipeline
    #Create ENV
    print("making")
    
    #overwriting certain variables to make testing easier
    num_procs = 1

    #env = make_multiprocess_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, seed, use_domain_rand, domain_rand_args,num_procs)
    #env = make_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, 6, seed=0, use_domain_rand=False, domain_rand_args=None)
    #env = SubprocVecEnv(env)
    env = make_multiprocess_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, seed, use_domain_rand, domain_rand_args,num_procs)
    env = VecTransposeImage(SubprocVecEnv(env))

    # Create model
    if policy == 'PPO':
       model = PPO(policy_type, env= env, **policy_kwargs)
       print("PPO")
    elif policy == 'SAC':
       model = SAC(policy_type, env = env, **policy_kwargs)
       print("SAC")
    else: 
       ("-----------ERRROR no policy selected------------")

    print("Created a new model")        
    

    # Create callback
    #env.training = False

    #eval_callback = CustomEvalCallback(env, **messages_to_eval_callback)
    #callback = CallbackList([eval_callback])

    #Settup tests!
    #print("Env action space", env.action_space)
    #print("Emv observation space", env.observation_space)
    
    obs = env.reset()

    #print(obs['custom_image_rgbd'][0][0][0].dtype)
    #print(obs['custom_image_rgbd'][0][0][3].dtype)


    # frame_rgb = obs['custom_image_rgbd'][:,:,:3]

    # frame_d = obs['custom_image_rgbd'][:,:,3]

    # crop image 
    # print nicely
    # see printing from machine learning
    #print(frame_rgb)

    #print(frame_d.shape)

    #img.save('my.png')

    # rgb_img = ndimage.rotate(frame_rgb, 180)
    # rgb_img = Image.fromarray(rgb_img, 'RGB')
    # rgb_img.save('rgb.png')
    # rgb_img.show()
    
    

    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.utils import obs_as_tensor
    #check_env(env)
    #obs_tens = obs_as_tensor(obs,model.device)
    #policy = model.policy
    #print(policy.extract_features(obs_tens))
    print(model.policy)


# heat map plot
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    from scipy import ndimage

    # cropped_d = frame_d[4:30,29:55]
    # #cropped_d = frame_d[4:14,45:55]
    # cropped_d = ndimage.rotate(cropped_d, 180)

    # d_img = Image.fromarray(cropped_d)
    # d_img.save('d.png')
    # d_img.show()

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(20,5))  
    # sns.heatmap(cropped_d, vmin = 88, vmax=110, annot=True, fmt='g')
    # plt.show()


