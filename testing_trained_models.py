import robosuite as suite
import gym
import numpy as np

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


filename = 'test'

#Testing
env_robo = GymWrapper(
        suite.make(
            env_name="Lift",
            robots = "IIWA",
            gripper_types="Robotiq85Gripper", 
            has_renderer = True,
            has_offscreen_renderer= False,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq = 20,
            render_camera = None,
            horizon = 200,
            reward_shaping = True,
        )
    )


# Load model
model = PPO.load('trained_models/' + filename)
# Load the saved statistics
env = DummyVecEnv([lambda : env_robo])
env = VecNormalize.load('trained_models/vec_normalize_' + filename + '.pkl', env)
#  do not update them at test time
env.training = False
# reward normalization
env.norm_reward = False

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env_robo.render()
    if done:
        obs = env.reset()