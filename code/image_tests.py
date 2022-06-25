from robosuite.wrappers import DomainRandomizationWrapper

from PIL import Image

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
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14, IIWA_14_modified, IIWA_14_modified_flange, IIWA_14_modified_flange_multi
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14, Robotiq85Gripper_iiwa_14_longer_finger
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.helper_functions.wrap_env import make_multiprocess_env, make_env
from src.helper_functions.camera_functions import adjust_width_of_image
from src.helper_functions.hyperparameters import linear_schedule_1,linear_schedule_2, linear_schedule_3
from src.helper_functions.customCombinedExtractor import MediumCombinedExtractor ,CustomCombinedExtractor, LargeCombinedExtractor, CustomCombinedExtractor_object_obs
from src.helper_functions.customCombinedSurreal import CustomCombinedSurreal



#temp
from src.wrapper.GymWrapper_multiinput_RGBD import GymWrapper_multiinput_RGBD
from src.wrapper.GymWrapper_multiinput import GymWrapper_multiinput
if __name__ == '__main__':
    register_robot(IIWA_14_modified)
    register_robot(IIWA_14)
    register_robot(IIWA_14_modified_flange)
    register_robot(IIWA_14_modified_flange_multi)
    register_gripper(Robotiq85Gripper_iiwa_14)
    register_gripper(Robotiq85Gripper_iiwa_14_longer_finger)
    register_robot_class_mapping("IIWA_14")
    register_robot_class_mapping("IIWA_14_modified")
    register_robot_class_mapping("IIWA_14_modified_flange")
    register_robot_class_mapping("IIWA_14_modified_flange_multi")
    register_env(Lift_edit)
    register_env(Lift_4_objects)
    register_env(Lift_edit_green)
    register_env(Lift_edit_multiple_objects)

    #yaml_file = "config_files/" + input("Which yaml file to load config from: ")
    yaml_file = "config_files/ppo_rgbd_final_multi.yaml"
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)



    # Environment specifications
    env_options = config["robosuite"]
    if env_options["custom_camera_conversion"]:
        env_options["camera_widths"] = adjust_width_of_image(env_options["camera_heights"])
    env_options["custom_camera_trans_matrix"] = np.array(env_options["custom_camera_trans_matrix"])
    env_id = env_options.pop("env_id")
    neg_rew = env_options['neg_rew']
    use_rgbd = env_options['use_rgbd']
    env_options.pop('use_rgbd')
    env_options.pop('neg_rew')


    domain_yaml_file = "config_files/" + env_options['domain_arg_yaml']
    with open(domain_yaml_file, 'r') as stream:
        domain_config = yaml.safe_load(stream)
    env_options.pop('domain_arg_yaml')

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
    close_img = obs_config['close_img']
    add_noise = obs_config['add_noise']

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

    #Implementing learning rate schedular if 
    if config["learning_rate_schedular"] == 1:
        policy_kwargs["learning_rate"] = linear_schedule_1(policy_kwargs["learning_rate"])
    elif config["learning_rate_schedular"] == 2:
        policy_kwargs["learning_rate"] = linear_schedule_2(policy_kwargs["learning_rate"])
    elif config["learning_rate_schedular"] == 3:
        policy_kwargs["learning_rate"] = linear_schedule_3(policy_kwargs["learning_rate"])
    



    #Implementing custom feature extractor
    if policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'large':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = LargeCombinedExtractor
    elif policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'small':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = CustomCombinedExtractor
    elif policy_kwargs["policy_kwargs"]["features_extractor_class"] == 'medium':
        policy_kwargs["policy_kwargs"]["features_extractor_class"] = MediumCombinedExtractor
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

    #Settings for wandb
    wandb_settings = config["wandb"]
    wandb_filename = config["wandb_filename"]

    # RL pipeline
    #Create ENV
    print("making")
    
    #env = VecTransposeImage(SubprocVecEnv([make_multiprocess_env(use_rgbd, env_id, env_options, obs_list, smaller_action_space, xyz_action_space,  i, seed, use_domain_rand=use_domain_rand, domain_rand_args=domain_rand_args) for i in range(num_procs)]))
    # from robosuite.renderers import load_renderer_config
    # ren_config = load_renderer_config("igibson")
    #env = GymWrapper_multiinput(suite.make(env_id, **env_options), obs_list, smaller_action_space, xyz_action_space, close_img, neg_rew, use_rgbd, add_noise)
    env = GymWrapper_multiinput_RGBD(suite.make(env_id, **env_options, renderer_config=config), obs_list, smaller_action_space, xyz_action_space, close_img, neg_rew, add_noise)
   # env = make_env(add_noise, use_rgbd, neg_rew, close_img, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, rank = 0, seed=0, use_domain_rand=False, domain_rand_args=None)
    
    seed = np.random.randint(0,1000)
    if use_domain_rand:
        env = DomainRandomizationWrapper(env, seed= seed, **domain_rand_args)
        
    
    
    
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    from scipy import ndimage

    obs = env.reset()

    action = [0,0,0,0]
    time = 10
    # for i in range(time):
    #     obs,reward,done,info = env.step(action)
    # # Setting up variables
    obs,reward,done,info = env.step(action)

    def add_precision_noice(input):
        noice = np. random. normal(0, 0.000055, input.shape)
        output = input + noice     # stochastic noice Search for function online
        return output

    def add_local_planarity_precision(input):
        noice = np. random. normal(0, 0.000075, input.shape)
        output = input + noice     # stochastic noice Search for function online
        return output

    def add_global_planarity_precision(input):
        noise = input * np. random. normal(0, 0.000160, 1)
        output = input + noise     # stochastic noice Search for function online
        return output

    def add_noise_func(input):
        input = add_global_planarity_precision(input)
        input = add_local_planarity_precision(input)
        output = add_precision_noice(input)
        return output



    image = obs['custom_image_rgbd']
    frame_rgb = image[:,:,:3]
    frame_d = image[:,:,3]

    print(obs['robot0_joint_pos'])

    image = obs['custom_image_rgbd']
    frame_rgb = image[:,:,:3]
    d_img = ndimage.rotate(frame_rgb, 180)
    d_img = Image.fromarray(d_img,'RGB')
    d_img.save('rgb.png')  
    print(env.camera_modder.get_fovy('custom'))


    obs = env.reset()
    obs,reward,done,info = env.step(action)
    image = obs['custom_image_rgbd']
    frame_rgb = image[:,:,:3]
    d_img = ndimage.rotate(frame_rgb, 180)
    d_img = Image.fromarray(d_img,'RGB')
    d_img.save('rgb2.png') 
    print(env.camera_modder.get_fovy('custom'))

    obs = env.reset()
    obs,reward,done,info = env.step(action)
    image = obs['custom_image_rgbd']
    frame_rgb = image[:,:,:3]
    d_img = ndimage.rotate(frame_rgb, 180)
    d_img = Image.fromarray(d_img,'RGB')
    d_img.save('rgb3.png') 
    print(env.camera_modder.get_fovy('custom'))



    #comp = frame_d - noise_d
    
    
    #np.save('robosuite_image.npy',image)

    #cropped_rgb = frame_rgb[:65,23:177,:]

    #print("robot joint position is", joint_pos)

#     # crop image 
#     # print nicely
#     # see printing from machine learning
#     #print(frame_rgb)

#     #print(frame_d.shape)

#     #img.save('my.png')

    # d_img = ndimage.rotate(frame_d, 180)
    # d_img = Image.fromarray(frame_d)
    # d_img.save('d.png')   

    # rgb_img = ndimage.rotate(frame_rgb, 180)
    # rgb_img = Image.fromarray(rgb_img, 'RGB')
    # rgb_img.save('rgb.png')    
    

#     from stable_baselines3.common.env_checker import check_env
#     from stable_baselines3.common.utils import obs_as_tensor
#     #check_env(env)
#     #obs_tens = obs_as_tensor(obs,model.device)
#     #policy = model.policy
#     #print(policy.extract_features(obs_tens))
#     #print(model.policy)


# # heat map plot
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    from scipy import ndimage

#     cropped_d = frame_d[4:30,29:55]
#     # #cropped_d = frame_d[4:14,45:55]
#     cropped_d = ndimage.rotate(cropped_d, 180)

#     # d_img = Image.fromarray(cropped_d)
#     # d_img.save('d.png')
#     # d_img.show()ndimage.rotate
    

#     frame_d = ndimage.rotate(frame_d, 180)
#     frame_d = Image.fromarray(frame_d)
#     frame_d.save('robosuite_d.png')

#     frame_rgb = ndimage.rotate(frame_rgb, 180)
#     frame_rgb = Image.fromarray(frame_rgb, 'RGB')
#     frame_rgb.save('robosuite_rgb.png')



    # cropped_d = ndimage.rotate(frame_d, 180)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(20,5))  
    # sns.heatmap(cropped_d, annot=True, fmt='g')   # vmin, vmax
    # plt.show()
    # plt.savefig('plot_d.png')
    # env.close()


# Loop reseting env multiple times
