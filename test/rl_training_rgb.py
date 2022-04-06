import robosuite as suite
import gym
import numpy as np

from src.environments import Lift_4_objects

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def wrap_env(env):
    wrapped_env = Monitor(env)                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    return wrapped_env

register_env(Lift_4_objects)

config = load_controller_config(default_controller="OSC_POSE")

print("Starting to wrap environment")

# Training
env = GymWrapper(
        suite.make(
            env_name="Lift_4_objects",
            robots = "IIWA",
            controller_configs = config, 
            gripper_types="Robotiq85Gripper",      
            has_renderer=False,                    
            has_offscreen_renderer=True,           
            control_freq=20,                       
            horizon=1000,                          
            use_object_obs=False,                  
            use_camera_obs=True,
	    camera_heights=48,
	    camera_widths=48,                   
        ), ["agentview_image"]
)

env = wrap_env(env)

filename = 'rgb_4_objects'

model = PPO('MlpPolicy', env, verbose=2, tensorboard_log='./ppo_lift_4_objects_tensorboard/')
print("starting to learn")
model.learn(total_timesteps= 25000, log_interval= 5000,  tb_log_name=filename)

print("finished learning")

model.save('trained_models/' + filename)
env.save('trained_models/vec_normalize_' + filename + '.pkl')     # Save VecNormalize statistics


