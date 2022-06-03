from grpc import ssl_server_certificate_configuration
import robosuite as suite

from robosuite.models.robots.robot_model import register_robot
from robosuite.wrappers import DomainRandomizationWrapper

from src.wrapper.GymWrapper_multiinput import GymWrapper_multiinput
from src.wrapper.GymWrapper_multiinput_RGBD import GymWrapper_multiinput_RGBD

from src.models.robots.manipulators.iiwa_14_robot import IIWA_14, IIWA_14_modified, IIWA_14_modified_flange
from src.models.grippers.robotiq_85_iiwa_14_gripper import Robotiq85Gripper_iiwa_14, Robotiq85Gripper_iiwa_14_longer_finger
from src.helper_functions.register_new_models import register_gripper, register_robot_class_mapping

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed



def make_multiprocess_env(env_id, env_options, obs_list, smaller_action_space, xyz_action_space, seed, use_domain_rand, domain_rand_args, close_img, neg_rew, num_procs):

    multi_process_env = [make_env(neg_rew, close_img, env_id, env_options, obs_list, smaller_action_space, xyz_action_space,  i, seed, use_domain_rand=use_domain_rand, domain_rand_args=domain_rand_args) for i in range(num_procs)]

    return multi_process_env

def make_env(neg_rew, close_img, env_id, env_options, obs_list, smaller_action_space, xyz_action_space, rank, seed=0, use_domain_rand=False, domain_rand_args=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param observations: (str) observations to use from environment
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    print(close_img)
    def _init():
        register_robot(IIWA_14)
        register_robot(IIWA_14_modified)
        register_robot(IIWA_14_modified_flange)
        register_gripper(Robotiq85Gripper_iiwa_14)
        register_gripper(Robotiq85Gripper_iiwa_14_longer_finger)
        register_robot_class_mapping("IIWA_14")
        register_robot_class_mapping("IIWA_14_modified")
        register_robot_class_mapping("IIWA_14_modified_flange")

        

        env = GymWrapper_multiinput_RGBD(suite.make(env_id, **env_options), obs_list, smaller_action_space, xyz_action_space, close_img, neg_rew)
            
        if use_domain_rand:
            env = DomainRandomizationWrapper(env, seed= seed + rank, **domain_rand_args)
            env = Monitor(env, info_keywords = ("is_success",))
        else:
            env = Monitor(env, info_keywords = ("is_success",))
            env.seed(seed + rank)

        return env
    set_random_seed(seed)
    return _init

#TODO burde jeg sette en bestemt seed på denne? Det vil sørge for at de samme resultatene vil komme hver 
#gang, men jeg kan være uheldig med valg av seed
def make_singel_env(env_id, options, observations, smaller_action_space):
    register_robot(IIWA_14)
    register_gripper(Robotiq85Gripper_iiwa_14)
    register_robot_class_mapping("IIWA_14")
    
    env = GymWrapper_multiinput(suite.make(env_id, **options), observations, smaller_action_space)
    env = Monitor(env, info_keywords = ("is_success",)) 
    env = DummyVecEnv([lambda : env])
    #env = VecTransposeImage(env)
    return env