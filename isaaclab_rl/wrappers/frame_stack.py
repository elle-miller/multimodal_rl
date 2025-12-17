"""Wrapper that stacks frames."""

import gymnasium as gym
import numpy as np
import torch
from collections import deque
from gymnasium.error import DependencyNotInstalled
from typing import Union


class LazyFrames:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """

    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames", "channels_first")

    def __init__(self, frames: list, lz4_compress: bool = False):
        """Lazyframe for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """

        self.frame_shape = tuple(frames[0].shape)

        if len(self.frame_shape) > 2:

            if np.argmin(self.frame_shape[1:]) != 0:
                self.channels_first = False
                self.shape = self.frame_shape[:-1] + (self.frame_shape[-1] * len(frames),)
            else:
                self.channels_first = True
                self.shape = (self.frame_shape[0] * len(frames),) + self.frame_shape[1:]
        else:
            self.shape = (len(frames),) + self.frame_shape
            self.channels_first = True

        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
            except ImportError as e:
                raise DependencyNotInstalled("lz4 is not installed, run `pip install gymnasium[other]`") from e

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.

        Args:
            dtype: The dtype of the stacked frames

        Returns:
            The array of stacked frames with dtype
        """
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        """Returns the number of frame stacks.

        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __getitem__(self, int_or_slice: Union[int, slice]):
        """Gets the stacked frames for a particular index or slice.

        This function is called when we are transforming the LazyTensor into an "active" Tensor

        Args:
            int_or_slice: Index or slice to get items for

        Returns:
            np.stacked frames for the int or slice

        """

        if isinstance(int_or_slice, int):
            assert int_or_slice <= len(self._frames), f"Given idx(s) in {int_or_slice} out-of-bounds."
            return self._check_decompress(self._frames[int_or_slice])  # single frame

        # Does this generalizes to both pixels and prop tensors, even when we want to framestack the props?
        return torch.concatenate(
            [self._check_decompress(x) for x in self._frames], dim=1 if self.channels_first else -1
        )

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `obs_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        obs_stack,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            obs_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(self, obs_stack=obs_stack, lz4_compress=lz4_compress)
        gym.ObservationWrapper.__init__(self, env)

        print("***USING FRAME STACK:", obs_stack)

        self.obs_stack = obs_stack
        self.lz4_compress = lz4_compress

        # We expect our environments to provide a dictionary of observation spaces of type `gym.spaces.Dict`
        # Also, we expect that the frame-stack computation should be handled *inside of the environment class*!!!
        # Finally, we are only framestacking the pixels at the moment. This can be changed easily here.
        self.frames = {}
        for k, v in self.observation_space["policy"].items():
            if k == "pixels" or k == "prop" or k == "tactile" or k == "gt":
                self.frames[k] = deque(maxlen=obs_stack)
            else:
                self.frames[k] = deque(maxlen=1)

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        obs = {}
        for k, v in self.frames.items():
            obs[k] = LazyFrames(list(v), self.lz4_compress)

        # return obs as policy
        obs_dict = {"policy": obs}

        if "aux" in observation.keys():
            aux_dict = {"aux": observation["aux"]}
            obs_dict.update(aux_dict)

        return obs_dict

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """

        # observation might have both "policy" and "aux", and we will want to
        observation, reward, terminated, truncated, info = self.env.step(action)

        for k, v in observation["policy"].items():
            self.frames[k].append(v)

        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def get_reset_obs(self, obs):
        """
        Note : current implementation only fills frames at the beginning
        
        """

        for k, v in obs["policy"].items():
            for _ in range(self.obs_stack):
                self.frames[k].append(v)
        obs = self.observation(obs)
        return obs
