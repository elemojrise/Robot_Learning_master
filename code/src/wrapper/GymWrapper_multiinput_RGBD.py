"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper
from robosuite.utils.camera_utils import get_real_depth_map
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


class GymWrapper_multiinput_RGBD(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, smaller_action_space = False, xyz_action_space = False, use_rgbd = False):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)


        self.smaller_action_space = smaller_action_space
        self.xyz_action_space = xyz_action_space

        assert keys is not None, (
            "You need to specifi which observation keys to use when using the CustomGymWrapper "
        )
        self.keys = keys

        
        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation

        #### Everything above is good
        obs = self.env.reset()
        temp_dict = {}

        print("----------------------Starting inniting")
        

        for key in self.keys:
            low = -np.inf
            high = np.inf
            dtype = np.float32
            if "rgbd" in key:
                low = 0
                high = 255
                dtype = np.uint8      ####Currently uint8
                shape = (obs[self.env.camera_names[0]+"_image"].shape)
                shape_list = list(shape)
                shape_list[2] = shape_list[2] + 1
                shape = shape_list
                temp_dict[self.env.camera_names[0] + "rgbd"] = spaces.Box(low = low,high = high, shape=shape,dtype= dtype)
            else:
                shape = (obs[key].shape)
                temp_dict[key] = spaces.Box(low = low,high = high, shape=shape,dtype= dtype)
        self.observation_space = spaces.Dict(temp_dict)
        
        #### Everyting below is good
        #Setting up action space
        #Changing the value of the action space
        low, high = self.env.action_spec
        if self.smaller_action_space:
            low, high = low[:-2], high[:-2] #trekker fra a og b som mulige inputs

        if self.xyz_action_space:
            low, high = low[:-1], high[:-1] #trekker fra c som mulige input

        self.action_space = spaces.Box(low=np.float32(low), high=np.float32(high))


        #variable for checking grasp sucess
        self.grasp_success = 0


        print("----------------------Donne inniting")

    def _multiinput_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """


        print("----------------------Running multiinputs_obs")
        ob_lst = {}
        for key in self.keys:
            if self.env.camera_names[0] in key:
                from scipy import ndimage
                from PIL import Image


                cam_name = self.env.camera_names[0]
                depth_array_normalized = obs_dict[cam_name +"_depth"]

                depth_map = np.uint8(np.clip(get_real_depth_map(self.sim, depth_array_normalized)*(255/3), 0,255))   ## maps from 0-3 to 0-255 and cuts all values over 255
                # print(depth_map.shape)
                # rgb_img = ndimage.rotate(depth_map, 180)
                # rgb_img = np.squeeze(rgb_img, axis=2) 
                # rgb_img = Image.fromarray(rgb_img)
                # rgb_img.show()
                depth_map = np.clip(get_real_depth_map(self.sim, depth_array_normalized)*(65535/3), 0,65535).astype(np.uint16)   #65535

                depth_array = depth_map
                rgb_array = obs_dict[cam_name + "_image"]

                new_array = np.concatenate((rgb_array, depth_array), axis=-1)

                ob_lst[key] = new_array

            elif key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst[key] =obs_dict[key]
        return ob_lst
    

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        self.grasp_success = 0

        ob_dict = self.env.reset()
        return self._multiinput_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        if self.smaller_action_space:
            action = np.insert(action, 3, 0)
            action = np.insert(action, 4, 0)

        if self.xyz_action_space:
            action = np.insert(action, 5, 0)    
        ob_dict, reward, done, info = self.env.step(action)

        

        # It will now keep being 1 until reset
        if self.env._check_success():
            print("succesful_grasp")
            self.grasp_success = 1
        
        info["is_success"] = self.grasp_success

        return self._multiinput_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
