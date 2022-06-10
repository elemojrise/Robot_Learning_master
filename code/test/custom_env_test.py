import numpy as np
import robosuite as suite
from PIL import Image

from scipy.spatial.transform import Rotation as R

from src.environments import Lift_4_objects, lift_edit

from robosuite.utils.camera_utils import CameraMover

from robosuite.environments.base import register_env

from src.helper_functions.camera_functions import adjust_width_of_image

from robosuite.models.robots.robot_model import register_robot

from src.wrapper import GymWrapper_multiinput
from src.models.robots.manipulators.iiwa_14_robot import IIWA_14
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv

#Registrerer custom environment
register_robot(IIWA_14)
register_gripper(Robotiq85Gripper_iiwa_14)
register_robot_class_mapping("IIWA_14")
register_env(Lift_4_objects)
register_env(lift_edit)

Trans_matrix = np.array([ [ 0.171221,  0.730116, -0.661524,  1124.551880], 
  [ 0.985078, -0.138769,  0.101808, -46.181087], 
  [-0.017467, -0.669085, -0.742981,  815.163208], 
  [ 0.000000,  0.000000,  0.000000,  1.000000] ])

Trans_matrix_20_points = np.array([ [ 0.011358,  0.433358, -0.901150,  1220.739746], 
  [ 0.961834,  0.241668,  0.128340, -129.767868], 
  [ 0.273397, -0.868215, -0.414073,  503.424103], 
  [ 0.000000,  0.000000,  0.000000,  1.000000] ])  

Trans_matrix_over_the_shoulder = np.array([ [ 0.477045, -0.841291,  0.254279,  251.447571], 
  [-0.877611, -0.471518,  0.086427, -159.115860], 
  [ 0.047187, -0.264388, -0.963261,  1172.085205], 
  [ 0.000000,  0.000000,  0.000000,  1.000000] ])

Ned_to_Enu_conversion_1 = np.array([  [1,0,0],
                                        [0,-1,0],
                                        [0,0,-1]
                        ])

Ned_to_Enu_conversion_2 = np.array([  [0,1,0],
                                        [1,0,0],
                                        [0,0,-1]
                        ])


# create environment instance
env = suite.make(
    env_name="Lift_edit",
    robots = "IIWA_14",
    gripper_types="Robotiq85Gripper_iiwa_14",                # use default grippers per robot arm
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=1000,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names="custom",
    render_camera = None,               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths= 84, #adjust_width_of_image(100),                       # image width
    # reward_shaping=False,
    custom_camera_name = "custom",
    custom_camera_trans_matrix = Trans_matrix,
    custom_camera_conversion= False,
    custom_camera_attrib = {"fovy": 36},
)

env = GymWrapper_multiinput(env=env, keys= ["custom_image"])
env = Monitor(env, info_keywords = ("is_success",)) 
env = DummyVecEnv([lambda : env])
# env = VecTransposeImage(env)

# cam_pose = np.add(cam_pose, env.table_offset)

# reset the environment
env.reset()

#action = np.random.randn(env.robots[0].dof) # sample random action
action = [env.action_space.sample()]
print(action)
obs, reward, done, info = env.step(action)  # take action in the environment

#print(obs)

# cam_move = CameraMover(env= env, camera = "custom", init_camera_pos= cam_pose , init_camera_quat= np.array([[0.653], [0.271], [0.271], [0.653]]).flatten() )

# pose, quat = cam_move.get_camera_pose()

#print(pose, quat)

image = obs['custom_image'] #uint8
image = np.squeeze(image)
print(image.shape)
img = Image.fromarray(image, 'RGB')
img = img.rotate(180)

rot_img = np.asarray(img)
img.save('20_points.png')


# print(obs)
# print(reward)
# print(done)
# print(info)

# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()

