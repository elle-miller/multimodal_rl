import gymnasium
import torch
from typing import Any, Tuple, Union

from skrl import config

"""
File from SKRL

"""


class IsaacLabWrapper(object):
    def __init__(self, env: Any, num_eval_envs, obs_stack: int = 1, debug: bool = False) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        # device
        if hasattr(self._unwrapped, "device"):
            self._device = config.torch.parse_device(self._unwrapped.device)
        else:
            self._device = config.torch.parse_device(None)

        self._observations = None
        self._info = {}
        self.obs_stack = obs_stack
        self.debug = debug
        self.first_reset = True
        self.num_eval_envs = num_eval_envs
        self.eval_env_ids = torch.arange(self.num_eval_envs, dtype=torch.int64, device=self.device)


    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    def get_observations(self):
        try:
            self._env.get_observations()
        except:
            self._unwrapped.get_observations()
        return


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._observations, reward, terminated, truncated, self._info = self._env.step(actions)

        if self.debug:
            for k, v in self._observations["policy"].items():
                # actviate the LazyFrame
                self._check_instability(v[:], f"observations_{k}")
            self._check_instability(actions, "actions")
            self._check_instability(reward, "reward")

        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self, hard=False) -> Tuple[torch.Tensor, Any]:
        """Reset the environmentc
        """
        # reset everything
        obs, self._info = self._env.reset()

        # if we are frame stacking need to duplicate start frames
        if hard and self.obs_stack != 1:
            print("[Hard reset] : Duplicating frames")
            obs = self.get_reset_obs(obs)

        self._observations = obs

        return self._observations, self._info
    
    def reset_eval_envs(self):
        self._unwrapped.scene.reset(self.eval_env_ids)
        self._unwrapped.episode_length_buf[self.eval_env_ids] = 0

        # update observations
        self._observations = self._unwrapped.get_observations()

        # convert to LazyFrames
        if self.obs_stack != 1:
            self._observations = self._env.observation(self._observations)

        return self._observations, self._info 
    

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()

    @property
    def device(self) -> torch.device:
        """The device used by the environment
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments
        """
        return self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return self._unwrapped.single_observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return self._unwrapped.single_action_space
    
    # Copied idea from Stable Baselines VecCheckNan thx Antonin :)
    def _check_instability(self, x, name):
        if torch.isnan(x).any():
            print(f"IsaacLabWrapper / {name} is nan", torch.isnan(x).any())
        if torch.isinf(x).any():
            print(f"IsaacLabWrapper / {name} is inf", torch.isinf(x).any())