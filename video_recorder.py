import robosuite as suite
import gym
import numpy as np

from environments import Lift_4_objects

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecNormalize

from stable_baselines3.common.evaluation import evaluate_policy

register_env(Lift_4_objects)

filename = 'rgb_4_objects'

config = load_controller_config(default_controller="OSC_POSE")


env_robo = GymWrapper(
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

# Load model
model = PPO.load('trained_models/' + filename)
# Load the saved statistics
env = Monitor(env_robo)
env = DummyVecEnv([lambda : env])
env = VecNormalize.load('trained_models/vec_normalize_' + filename + '.pkl', env)
#  do not update them at test time
env.training = False
# reward normalization
env.norm_reward = False

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()




