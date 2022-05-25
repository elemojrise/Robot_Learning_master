import gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict


class LargeCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(LargeCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = LargeCNN(subspace, features_dim=cnn_output_dim) # Denne sier størrelsen på linear layer
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size


    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            to_extractor = observations[key]
            if len(to_extractor.shape) == 1:
                to_extractor = to_extractor.unsqueeze(1)
            encoded_tensor_list.append(extractor(to_extractor))

        return th.cat(encoded_tensor_list, dim=1)



class LargeCNN(BaseFeaturesExtractor):
    """

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #tror feature dim er det de refererer til som size
        super(LargeCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=6, stride=2),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))





class CustomCombinedExtractor(BaseFeaturesExtractor):


    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = CustomNatureCNN(subspace, features_dim=cnn_output_dim) # Denne sier størrelsen på linear layer
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        # self.rnn_stem = nn.LSTM(input_size = total_concat_size,
        #                         hidden_size = 256, #Det er dette Surreal bruker
        #                         num_layers = 1,
        #                         batch_first=True)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            to_extractor = observations[key]
            if len(to_extractor.shape) == 1:
                to_extractor = to_extractor.unsqueeze(1)
            encoded_tensor_list.append(extractor(to_extractor))

        return th.cat(encoded_tensor_list, dim=1)


class CustomNatureCNN(BaseFeaturesExtractor):
    """

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #tror feature dim er det de refererer til som size
        super(CustomNatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



class CustomCombinedExtractor_object_obs(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").
    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCombinedExtractor_object_obs, self).__init__(observation_space, features_dim=1)

        extractors = {}


        self.rest_key = "rest"

        self.not_image_key_list = []
        total_concat_size = 0
        total_flat_concat_size = 0
        output_size_linear = 64

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = CustomNatureCNN_object_obs(subspace, features_dim=cnn_output_dim) # Denne sier størrelsen på linear layer
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                self.not_image_key_list.append(key)
                total_flat_concat_size += get_flattened_obs_dim(subspace)

        
        extractors[self.rest_key] = nn.Sequential(
            nn.Linear(total_flat_concat_size, 256),
            nn.Tanh(),
            nn.Linear(256, output_size_linear),
            nn.Tanh()
        )
        
        
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size + output_size_linear

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        not_image_list = []
        for key in self.not_image_key_list:
            to_list = observations[key]
            if len(to_list.shape) == 1:
                to_list = to_list.unsqueeze(1)
            not_image_list.append(to_list)
        
        not_image_list = th.cat(not_image_list, dim=1)

        for key, extractor in self.extractors.items():
            if key == self.rest_key:
                encoded_tensor_list.append(extractor(not_image_list))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
    
        return th.cat(encoded_tensor_list, dim=1)


class CustomNatureCNN_object_obs(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #tror feature dim er det de refererer til som size
        super(CustomNatureCNN_object_obs, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=6, stride=2, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
