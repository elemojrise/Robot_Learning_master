import numpy as np
import robosuite as suite
from PIL import Image

from scipy.spatial.transform import Rotation as R

from src.environments import Lift_4_objects, lift_edit

from robosuite.utils.camera_utils import CameraMover

from robosuite.environments.base import register_env

#Registrerer custom environment
register_env(Lift_4_objects)
register_env(lift_edit)

Trans_matrix = np.array([ [ 0.171221,  0.730116, -0.661524,  1124.551880], 
  [ 0.985078, -0.138769,  0.101808, -46.181087], 
  [-0.017467, -0.669085, -0.742981,  815.163208], 
  [ 0.000000,  0.000000,  0.000000,  1.000000] ])

Ned_to_Enu_conversion_1 = np.array([  [1,0,0],
                                        [0,-1,0],
                                        [0,0,-1]
                        ])

Ned_to_Enu_conversion_2 = np.array([  [0,1,0],
                                        [1,0,0],
                                        [0,0,-1]
                        ])



def Rot_z(angle):
    return np.array([   [np.cos(angle),-np.sin(angle),0],
                        [np.sin(angle),np.cos(angle),0],
                        [0,0,1]])
def Rot_y(angle):
    return np.array([   [np.cos(angle),0, np.sin(angle)],
                        [0,1,0],
                        [-np.sin(angle),0, np.cos(angle)]])


def Rot_x(angle):
    return np.array([   [1,0,0],
                        [0, np.cos(angle),-np.sin(angle)],
                        [0, np.sin(angle),np.cos(angle)]])


stand_rot = Rot_z(np.pi/2)@Rot_x(np.pi/4)

# print(stand_rot)


cam_pose = Trans_matrix[:3,3]/1000 #converting to meter

correct_rotation = Trans_matrix[:3,:3]@Ned_to_Enu_conversion_1
print(correct_rotation)

cam_rot = R.from_matrix(correct_rotation).as_quat()[[3, 0, 1, 2]]

cam_rot = R.from_matrix(stand_rot).as_quat()[[3, 0, 1, 2]]



print(cam_rot)

legit = np.array([0.653, 0.271, 0.271, 0.653])


# create environment instance
env = suite.make(
    env_name="Lift_edit",
    robots = "IIWA",
    gripper_types="Robotiq85Gripper",                # use default grippers per robot arm
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=1000,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names="custom",
    render_camera = None,               # use "agentview" camera for observations
    camera_heights=1200,                      # image height
    camera_widths=1200,                       # image width
    # reward_shaping=False,
    custom_camera_name = "custom",
    custom_camera_trans_matrix = Trans_matrix,
    custom_camera_conversion= True,
    custom_camera_attrib = None,

)

# cam_pose = np.add(cam_pose, env.table_offset)

# reset the environment
env.reset()

action = np.random.randn(env.robots[0].dof) # sample random action
obs, reward, done, info = env.step(action)  # take action in the environment

# cam_move = CameraMover(env= env, camera = "custom", init_camera_pos= cam_pose , init_camera_quat= np.array([[0.653], [0.271], [0.271], [0.653]]).flatten() )

# pose, quat = cam_move.get_camera_pose()

#print(pose, quat)

image = obs['custom_image'] #uint8
img = Image.fromarray(image, 'RGB')
#img = img.rotate(180)
img.save('new_class_test.png')


# print(obs)
# print(reward)
# print(done)
# print(info)

# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()

