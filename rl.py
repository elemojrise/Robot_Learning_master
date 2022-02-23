import robosuite as suite
import os
import yaml

from wrapper import GymWrapper_rgb, GymWrapper_multiinput
from robosuite.environments.base import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


from environments import Lift_4_objects
from callback import ProgressBarManager


def make_robosuite_env(env_id, options, observations, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param observations: (str) observations to use from environment
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper_multiinput(suite.make(env_id, **options), observations)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == '__main__':
    register_env(Lift_4_objects)

    with open("rl_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Environment specifications
    env_options = config["robosuite"]
    env_id = env_options.pop("env_id")

    # Observations
    obs_config = config["observations"]
    obs_list = [obs_config["rgb"]] #lager en liste av det

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    check_pt_interval = sb_config["check_pt_interval"]
    num_procs = sb_config["num_procs"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    tb_log_folder = file_handling["tb_log_folder"]
    tb_log_name = file_handling["tb_log_name"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    load_model_filename = file_handling["load_model_filename"]

    continue_training_model_folder = file_handling["continue_training_model_folder"]
    continue_training_model_filename = file_handling["continue_training_model_filename"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    # Settings for pipeline
    training = config["training"]
    seed = config["seed"]

    # RL pipeline
    if training:
        if num_procs == 1:
            env = GymWrapper_multiinput(suite.make(env_id, **env_options), obs_list)
        else:
            env = SubprocVecEnv([make_robosuite_env(env_id, env_options, obs_list, i, seed) for i in range(num_procs)])

        # Create callback
        checkpoint_callback = CheckpointCallback(save_freq=check_pt_interval, save_path='./checkpoints/', 
                                name_prefix=save_model_filename, verbose=2)
        
        # Train new model
        if continue_training_model_filename is None:

            # Normalize environment
            #env = VecNormalize(env)

            "TODO Her må jeg legge inn policy_kwargs slik at det er mulig å lage eget nettverk"
            # Create model
            model = PPO(policy_type, env, tensorboard_log=tb_log_folder, verbose=1)

            print("Created a new model")

        # Continual training
        else:

            # Join file paths
            continue_training_model_path = os.path.join(continue_training_model_folder, continue_training_model_filename)
            continue_training_vecnormalize_path = os.path.join(continue_training_model_folder, 'vec_normalize_' + continue_training_model_filename + '.pkl')

            print(f"Continual training on model located at {continue_training_model_path}")

            # Load normalized env 
            env = VecNormalize.load(continue_training_vecnormalize_path, env)

            # Load model
            model = PPO.load(continue_training_model_path, env=env)

        # Training
        model.learn(total_timesteps=training_timesteps, tb_log_name=tb_log_name, callback=checkpoint_callback, reset_num_timesteps=True)

        # Save trained model
        model.save(save_model_path)
        env.save(save_vecnormalize_path)

    else:
        # Create evaluation environment
        env_options['has_renderer'] = True
        env_gym = GymWrapper(suite.make(env_id, **env_options))
        env = DummyVecEnv([lambda : env_gym])

        # Load normalized env
        env = VecNormalize.load(load_vecnormalize_path, env)

        # Turn of updates and reward normalization
        env.training = False
        env.norm_reward = False

        # Load model
        model = PPO.load(load_model_path, env)

        # Simulate environment
        obs = env.reset()
        eprew = 0
        while True:
            action, _states = model.predict(obs)
            print(f"action: {action}")
            obs, reward, done, info = env.step(action)
            #print(action)
            print(f'reward: {reward}')
            eprew += reward
            env_gym.render()
            if done:
                print(f'eprew: {eprew}')
                obs = env.reset()
                eprew = 0

        env.close()