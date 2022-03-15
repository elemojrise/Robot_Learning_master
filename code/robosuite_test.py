import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift",
    robots = "IIWA",
    gripper_types="Robotiq85Gripper",                # use default grippers per robot arm
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=False,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=False,                   # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,

)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render() 