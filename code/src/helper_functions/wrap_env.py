import robosuite as suite

from src.wrapper import GymWrapper_multiinput

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed




def make_multiprocess_env(env_id, options, observations, smaller_action_space, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param observations: (str) observations to use from environment
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper_multiinput(suite.make(env_id, **options), observations, smaller_action_space)
        env = Monitor(env, info_keywords = ("is_success",)) 
        env = VecTransposeImage(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

#TODO burde jeg sette en bestemt seed på denne? Det vil sørge for at de samme resultatene vil komme hver 
#gang, men jeg kan være uheldig med valg av seed
def make_singel_env(env_id, options, observations, smaller_action_space):
    env = GymWrapper_multiinput(suite.make(env_id, **options), observations, smaller_action_space)
    env = Monitor(env, info_keywords = ("is_success",)) 
    env = DummyVecEnv([lambda : env])
    env = VecTransposeImage(env)
    return env