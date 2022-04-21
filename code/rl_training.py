from curses import flash
import robosuite as suite
import gym
import numpy as np

from robosuite.models.robots.robot_model import register_robot
from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping
from src.wrapper.GymWrapper_multiinput import GymWrapper_multiinput
from src.helper_functions.hyperparameters import linear_schedule
from src.helper_functions.wrap_env import make_multiprocess_env
from src.helper_functions.customCombinedExtractor import CustomCombinedExtractor

from src.environments import Lift_4_objects, Lift_edit
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback

print("doing")
def wrap_env(env):
    wrapped_env = Monitor(env, info_keywords = ("is_success",))                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = SubprocVecEnv([wrapped_env])
    #wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    #wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    wrapped_env = VecTransposeImage(wrapped_env)
    return wrapped_env

controller_config = load_controller_config(default_controller="OSC_POSE")

register_robot(IIWA_14)
register_gripper(Robotiq85Gripper_iiwa_14)
register_robot_class_mapping("IIWA_14")
register_env(Lift_edit)

# Training
env = GymWrapper_multiinput(
        suite.make(
            env_name="Lift_edit",
            robots = "IIWA_14",
            controller_configs = controller_config, 
            gripper_types="Robotiq85Gripper_iiwa_14",      
            has_renderer=False,                    
            has_offscreen_renderer=True,           
            control_freq=10,                       
            horizon= 4,
            ignore_done = False, 
            camera_heights = 48,
            camera_widths = 48,                          
            use_object_obs=False,                  
            use_camera_obs=True,
            reward_shaping= True,
            #camera_names = ["all-robotview"]                   
        ),  
        keys = ["agentview_image","robot0_eef_pos", 'robot0_gripper_qpos'],
        #smaller_action_space= True
)
env = wrap_env(env)

# print(env.metadata)
# print(env.get_attr("metadata")[0])

# temp_env = env.venv
# print(temp_env)

# print(temp_env.metadata)
# print(temp_env.get_attr("metadata")[0])

# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 300 == 0, video_length=200)

# eval_callback = EvalCallback(env, callback_on_new_best=None, #callback_after_eval=None, 
#                             n_eval_episodes=3, eval_freq=200, log_path='./logs/', 
#                             best_model_save_path='best_model/logs/', deterministic=False, render=False, 
#                             verbose=1, warn=True)
# filename = 'test'

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(pi=[300, 200], vf=[300, 200])]
)

obs = env.reset()

model = PPO('MultiInputPolicy', env, policy_kwargs = policy_kwargs, n_steps = 8, batch_size= 2, verbose=1, device= "auto")
print("starting to learn")

model.learn(total_timesteps=40)

print("finished learning")

# model.save('trained_models/' + filename)
# env.save('trained_models/vec_normalize_' + filename + '.pkl')     # Save VecNormalize statistics


