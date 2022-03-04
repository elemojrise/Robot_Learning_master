import gym

from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

import robosuite as suite
from robosuite.environments.base import register_env
from robosuite import load_controller_config

from environments import Lift_4_objects
from wrapper import GymWrapper_multiinput, GymWrapper_rgb


register_env(Lift_4_objects)

config = {
    "policy_type": 'CnnPolicy',
    "total_timesteps": 25000,
}

run = wandb.init(
    project="sb3_lift",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

controller_config = load_controller_config(default_controller="OSC_POSE")

env = GymWrapper_rgb(
        suite.make(
            env_name="Lift",
            robots = "IIWA",
            controller_configs = controller_config, 
            gripper_types="Robotiq85Gripper",      
            has_renderer=False,                    
            has_offscreen_renderer=True,           
            control_freq=20,                       
            horizon=1000,
            camera_heights = 512,
            camera_widths = 512,                          
            use_object_obs=False,                  
            use_camera_obs=True,
            reward_shaping= True,
            #camera_names = ["all-robotview"]                   
        ),  
        keys = ["agentview_image"], #, "robot0_joint_pos_cos"],
        #smaller_action_space= True
)

obs = env.reset()

# img = Image.fromarray(obs, 'RGB')
# img.save('lift.png')

#env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()

