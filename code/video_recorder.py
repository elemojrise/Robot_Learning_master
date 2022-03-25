import robosuite as suite
import numpy as np
import imageio

import robosuite.utils.macros as macros
from robosuite.models.robots.robot_model import register_robot

from src.environments import Lift_4_objects, Lift_edit
from src.wrapper import GymWrapper_multiinput
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping

from robosuite.environments.base import register_env
from robosuite import load_controller_config

from stable_baselines3 import PPO

register_robot(IIWA_14)
register_gripper(Robotiq85Gripper_iiwa_14)
register_robot_class_mapping("IIWA_14")
register_env(Lift_4_objects)
register_env(Lift_edit)

config = load_controller_config(default_controller="OSC_POSE")


env_robo = GymWrapper_multiinput(
                suite.make(
                  env_name="Lift_edit",
                  robots = "IIWA_14",
                  controller_configs = config, 
                  gripper_types="Robotiq85Gripper_iiwa_14",      
                  has_renderer=False,                    
                  has_offscreen_renderer=True,           
                  control_freq=20,                       
                  horizon=400,                          
                  use_object_obs=False,                  
                  use_camera_obs=True,
                  camera_heights=300,
                  camera_widths=486,
                  camera_names = "custom",
                  custom_camera_name = "custom", 
                  custom_camera_trans_matrix = np.array([ [ 0.011358,  0.433358, -0.901150,  1220.739746], 
                                                [ 0.961834,  0.241668,  0.128340, -129.767868], 
                                                [ 0.273397, -0.868215, -0.414073,  503.424103], 
                                                [ 0.000000,  0.000000,  0.000000,  1.000000] ]),
                  custom_camera_conversion= True,
                  custom_camera_attrib=  {"fovy": 36}                   
            ), ["custom_image"]
)


def record_video(env, model, video_length, video_folder):
  macros.IMAGE_CONVENTION = "opencv"

  obs = env.reset()

  # create a video writer with imageio
  writer = imageio.get_writer(video_folder, fps=20)

  frames = []
  for i in range(video_length):

      action = model.predict(obs)
      obs, reward, done, info = env.step(action)

      frame = obs["custom" + "_image"]
      writer.append_data(frame)
      print("Saving frame #{}".format(i))

      if done:
        break

  writer.close()

model = PPO.load("/home/ojrise/best_model")
model.device = "cpu"


record_video(env=env_robo, model=model, video_length=400, video_folder="trying_to_make_video.mp4")

