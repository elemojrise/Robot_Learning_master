import robosuite as suite
import numpy as np

from src.environments import Lift_4_objects, Lift_edit

from robosuite.environments.base import register_env
from robosuite import load_controller_config
from src.wrapper import GymWrapper_rgb, GymWrapper_multiinput

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14

from robosuite.models.robots.robot_model import register_robot
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping

register_robot(IIWA_14)
register_gripper(Robotiq85Gripper_iiwa_14)
register_robot_class_mapping("IIWA_14")
register_env(Lift_edit)

def wrap_env(env):
    wrapped_env = Monitor(env)                          # Needed for extracting eprewmean and eplenmean
    wrapped_env = DummyVecEnv([lambda : wrapped_env])   # Needed for all environments (e.g. used for mulit-processing)
    wrapped_env = VecNormalize(wrapped_env)             # Needed for improving training when using MuJoCo envs?
    return wrapped_env



config = load_controller_config(default_controller="OSC_POSE")

env = GymWrapper_multiinput(
        suite.make(
            env_name="Lift_edit",
            robots = "IIWA_14",
            controller_configs = config, 
            gripper_types="Robotiq85Gripper_iiwa_14",      
            has_renderer=True,                    
            has_offscreen_renderer=False,           
            control_freq=20,                       
            horizon=10000,
            camera_heights = 48,
            camera_widths = 48,                          
            use_object_obs=False,                  
            use_camera_obs=False,                   
        ), ["robot0_joint_pos_cos"]
)

#env = wrap_env(env)

#Denne koden sjekker om environmentet er godkjent for å trene med stable_baseline
check_env(env)

#print("Getting observations")

obs = env.reset()

for i in range(10000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if done:
        env.reset()


#image = np.reshape(obs, (256,256,3))
#print("Observation = {} \n\n Action = {} \n\n".format(obs,action))

# model = PPO('MultiInputPolicy', env, verbose=2, tensorboard_log='./ppo_lift_4_objects_tensorboard/')

# model.learn(total_timesteps= 25000)

