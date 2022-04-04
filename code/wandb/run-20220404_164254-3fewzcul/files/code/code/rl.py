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
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from src.environments import Lift_4_objects, Lift_edit
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.helper_functions.wrap_env import make_multiprocess_env, make_singel_env
from src.helper_functions.camera_functions import adjust_width_of_image



if __name__ == '__main__':
    register_robot(IIWA_14)
    register_gripper(Robotiq85Gripper_iiwa_14)
    register_robot_class_mapping("IIWA_14")
    register_env(Lift_edit)
    register_env(Lift_4_objects)

    with open("rl_oj.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Environment specifications
    env_options = config["robosuite"]
    env_options["camera_widths"] = adjust_width_of_image(env_options["camera_heights"])
    env_options["custom_camera_trans_matrix"] = np.array(env_options["custom_camera_trans_matrix"])
    env_id = env_options.pop("env_id")

    #normalize obs and rew
    
    normalize_obs = config['normalize_obs']
    normalize_rew = config['normalize_rew']
    norm_obs_keys = config['norm_obs_keys']


    # Observations
    obs_config = config["gymwrapper"]
    obs_list = obs_config["observations"] 
    smaller_action_space = obs_config["smaller_action_space"]

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    check_pt_interval = sb_config["check_pt_interval"]
    num_procs = sb_config["num_procs"]
    model_type = sb_config['policy']

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

    #Settings for wandb
    wandb_settings = config["wandb"]



    # RL pipeline
    if training:
        if num_procs == 1:
            env = make_singel_env(env_id, env_options, obs_list, smaller_action_space)
        else:
            print("making")
            env = VecTransposeImage(SubprocVecEnv([make_multiprocess_env(env_id, env_options, obs_list, smaller_action_space,  i, seed) for i in range(num_procs)]))

    
        run = wandb.init(
            **wandb_settings,
            config=config,
        )
        print(env)

        # Create callback
        wandb_callback = WandbCallback(**messages_to_wand_callback, model_save_path=f"models/{run.id}")
        eval_callback = EvalCallback(env, **messages_to_eval_callback)
        callback = CallbackList([wandb_callback, eval_callback])
        
        # Train new model
        if continue_training_model_filename is None:

            if normalize_obs or normalize_rew:
                env = VecNormalize(env, norm_obs=normalize_obs,norm_reward=normalize_rew,norm_obs_keys=norm_obs_keys)
            # Create model
            model = model_type(policy_type, env= env, **policy_kwargs, tensorboard_log=f"runs/{run.id}")

            print("Created a new model")

        # Continual training
        else:

            # Join file paths
            continue_training_model_path = os.path.join(continue_training_model_folder, continue_training_model_filename)
            continue_training_vecnormalize_path = os.path.join(continue_training_model_folder, 'vec_normalize_' + continue_training_model_filename + '.pkl')

            print(f"Continual training on model located at {continue_training_model_path}")

            # Load normalized env
            if normalize_obs or normalize_rew:
                env = VecNormalize.load(continue_training_vecnormalize_path, env)

            # Load model
            model = model_type.load(continue_training_model_path, env=env)
            
        
        # Training
        print("starting to train")
        model.learn(total_timesteps=training_timesteps, callback=callback)

        run.finish()

        # Save trained model
        model.save(save_model_path)
        if normalize_obs or normalize_rew:
            env.save(save_vecnormalize_path)

        env.close()

    else:
        # Create evaluation environment
        env_options['has_renderer'] = True
        env_gym = GymWrapper(suite.make(env_id, **env_options))
        env = DummyVecEnv([lambda : env_gym])

        # Load normalized env
        env = VecNormalize.load(load_vecnormalize_path, env)

        # Turn of updates and reward normalization
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(load_model_path, env)

        # Simulate environment
        obs = env.reset()
        eprew = 0
        while True:
            action, _states = model.predict(obs)
            print(f"action: {action}")
            obs, reward, done, info = env.step(action)
            #print(action)
            print(f'reward: {reward}')
            eprew += reward
            env_gym.render()
            if done:
                print(f'eprew: {eprew}')
                obs = env.reset()
                eprew = 0

        env.close()
