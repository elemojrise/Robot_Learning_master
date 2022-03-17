from curses import flash
import robosuite as suite
import gym
import numpy as np

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper
from src.wrapper.GymWrapper_multiinput import GymWrapper_multiinput

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback

print("doing")
def wrap_env(env):
    wrapped_env = Monitor(env, info_keywords = ("is_success",))                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    #wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    wrapped_env = VecTransposeImage(wrapped_env)
    return wrapped_env

controller_config = load_controller_config(default_controller="OSC_POSE")

# Training
env = GymWrapper_multiinput(
        suite.make(
            env_name="Lift",
            robots = "IIWA",
            controller_configs = controller_config, 
            gripper_types="Robotiq85Gripper",      
            has_renderer=False,                    
            has_offscreen_renderer=True,           
            control_freq=20,                       
            horizon= 100,
            ignore_done = False, 
            camera_heights = 48,
            camera_widths = 48,                          
            use_object_obs=False,                  
            use_camera_obs=True,
            reward_shaping= True,
            #camera_names = ["all-robotview"]                   
        ),  
        keys = ["agentview_image"]#,"robot0_joint_pos"],
        #smaller_action_space= True
)
env = wrap_env(env)

# print(env.metadata)
# print(env.get_attr("metadata")[0])

# temp_env = env.venv
# print(temp_env)

# print(temp_env.metadata)
# print(temp_env.get_attr("metadata")[0])

env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 300 == 0, video_length=200)

eval_callback = EvalCallback(env, callback_on_new_best=None, #callback_after_eval=None, 
                            n_eval_episodes=3, eval_freq=200, log_path='./logs/', 
                            best_model_save_path='best_model/logs/', deterministic=False, render=False, 
                            verbose=1, warn=True)
filename = 'test'

obs = env.reset()

print(obs)

model = PPO('MultiInputPolicy', env, n_steps = 400, verbose=1)
print("starting to learn")

model.learn(total_timesteps=12000, callback = eval_callback)

print("finished learning")

# model.save('trained_models/' + filename)
# env.save('trained_models/vec_normalize_' + filename + '.pkl')     # Save VecNormalize statistics


