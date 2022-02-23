import numpy as np
import robosuite as suite
from PIL import Image

from environments import Lift_4_objects

from robosuite.environments.base import register_env

#Registrerer custom environment
register_env(Lift_4_objects)

# create environment instance
env = suite.make(
    env_name="Lift_4_objects",
    robots = "IIWA",
    gripper_types="Robotiq85Gripper",                # use default grippers per robot arm
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=1000,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    # camera_names="agentview",               # use "agentview" camera for observations
    # camera_heights=84,                      # image height
    # camera_widths=84,                       # image width
    # reward_shaping=False,

)

# reset the environment
env.reset()

action = np.random.randn(env.robots[0].dof) # sample random action
obs, reward, done, info = env.step(action)  # take action in the environment

print(obs)
#image = obs['agentview_image'] #uint8
#print(image.shape)

# print(obs)
# print(reward)
# print(done)
# print(info)

# for i in range(500):
#     action = np.random.randn(env.robots[0].dof) # sample random action
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     env.render()

# img = Image.fromarray(image, 'RGB')
# img.save('custom_env.png')
# img.show()