import gym
import gymnasium
import numpy as np
import torch
from torch import nn
from torch.utils.data import RandomSampler  # , BatchSampler
from typing import List, Optional, Tuple, Union

import kornia

from isaaclab_rl.wrappers.frame_stack import LazyFrames


class Memory:
    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        env_cfg: dict = {},
    ) -> None:
        """Base class representing a memory with circular buffers

        Buffers are torch tensors with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
                            (related to rollout length of e.g., PPO agent)
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional

        :raises ValueError: The export format is not supported
        """
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )

        # internal variables
        self.filled = False
        self.memory_index = 0

        self.total_samples = 0

        self.tensors = {}
        self.tensors_view = {}
        self.tensors_keep_dimensions = {}

        self.sampling_indexes = None
        self.all_sequence_indexes = np.concatenate(
            [np.arange(i, memory_size * num_envs + i, num_envs) for i in range(num_envs)]
        )

        # exporting data
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory

        if not self.export_format in ["pt", "np", "csv"]:
            raise ValueError(f"Export format not supported ({self.export_format})")

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned

        :return: Valid size
        :rtype: int
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs

    def share_memory(self) -> None:
        """Share the tensors between processes (irrelevant for GPU)"""
        for tensor in self.tensors.values():
            if not tensor.is_cuda:
                tensor.share_memory_()

    def get_tensor_names(self) -> Tuple[str]:
        """Get the name of the internal tensors in alphabetical order

        :return: Tensor names without internal prefix (_tensor_)
        :rtype: tuple of strings
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> torch.Tensor:
        """Get a tensor by its name

        :param name: Name of the tensor to retrieve
        :type name: str
        :param keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: ``True``)
                        If False, the returned tensor will have a shape of (memory size * number of environments, size)
        :type keepdim: bool, optional

        :raises KeyError: The tensor does not exist

        :return: Tensor
        :rtype: torch.Tensor
        """
        return self.tensors[name] if keepdim else self.tensors_view[name]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Set a tensor by its name

        :param name: Name of the tensor to set
        :type name: str
        :param tensor: Tensor to set
        :type tensor: torch.Tensor

        :raises KeyError: The tensor does not exist
        """
        with torch.no_grad():
            self.tensors[name].copy_(tensor)

    def _get_space_size(
        self, space: Union[int, Tuple[int], gym.Space, gymnasium.Space], keep_dimensions: bool = False
    ) -> Union[Tuple, int]:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param keep_dimensions: Whether or not to keep the space dimensions (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: If the space is not supported

        :return: Size of the space. If ``keep_dimensions`` is True, the space size will be a tuple
        :rtype: int or tuple of int
        """
        # I mean... really?
        if type(space) in [int, float]:
            return (int(space),) if keep_dimensions else int(space)
        elif type(space) in [tuple, list]:
            return tuple(space) if keep_dimensions else np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return (1,) if keep_dimensions else 1
            elif issubclass(type(space), gym.spaces.MultiDiscrete):
                return space.nvec.shape[0]
            elif issubclass(type(space), gym.spaces.Box):
                return tuple(space.shape) if keep_dimensions else np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                if keep_dimensions:
                    raise ValueError("keep_dimensions=True cannot be used with Dict spaces")
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        elif issubclass(type(space), gymnasium.Space):
            if issubclass(type(space), gymnasium.spaces.Discrete):
                return (1,) if keep_dimensions else 1
            elif issubclass(type(space), gymnasium.spaces.MultiDiscrete):
                return space.nvec.shape[0]
            elif issubclass(type(space), gymnasium.spaces.Box):
                return tuple(space.shape) if keep_dimensions else np.prod(space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                if keep_dimensions:
                    raise ValueError("keep_dimensions=True cannot be used with Dict spaces")
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        raise ValueError(f"Space type {type(space)} not supported")

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gym.Space, gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = True,
    ) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for sequences or gym/gymnasium spaces
        :type size: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param dtype: Data type (torch.dtype) (default: ``None``).
                      If None, the global default torch data type will be used
        :type dtype: torch.dtype or None, optional
        :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        size = self._get_space_size(size, keep_dimensions)
        # print(size)

        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError(f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})")
            return False

        # define tensor shape
        tensor_shape = (
            (self.memory_size, self.num_envs, *size) if keep_dimensions else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)

        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))

        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def reset(self) -> None:
        """Reset the memory by cleaning internal indexes and flags

        Old data will be retained until overwritten, but access through the available methods will not be guaranteed

        Default values of the internal indexes and flags

        - filled: False
        - env_index: 0
        - memory_index: 0
        """
        print("################################# MEMORY RESET ############################### ")
        self.filled = False
        self.memory_index = 0

    def add_samples(self, type, **tensors: torch.Tensor) -> None:
        """Record samples in memory


        :raises ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        if type == "sequential":
            self.add_sequential_samples(tensors)
        elif type == "parallel":
            self.add_parallel_samples(tensors)
            self.memory_index += 1
        else:
            raise ValueError(f"Unsupported type {type}")

        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True
            # export tensors to file
            # if self.export:
            #     self.save(directory=self.export_directory, format=self.export_format)

    def add_sequential_samples(self, tensors):

        # note this method assumes we will always have "prop" obs
        obs = tensors["states"]["prop"]

        # we do NOT save as LazyFrames [samples are already filtered by this point]
        if isinstance(obs, LazyFrames):
            raise TypeError("sort out LazyFrames before this please")

        if len(obs.shape) == 3:
            num_incoming_samples, seq_length, size = obs.shape
        elif len(obs.shape) == 2:
            num_incoming_samples, size = obs.shape
        else:
            raise ValueError

        self.total_samples += num_incoming_samples

        num_samples = min(num_incoming_samples, self.memory_size - self.memory_index)
        overflow_samples = num_incoming_samples - num_samples

        # print(f"adding {num_samples} seq to a memory index {self.memory_index}. overflow_samples {overflow_samples}") #, remaining_samples)

        # names are [states, actions, next_states] for the aux samples
        for name, tensor in tensors.items():
            # "states" is dicts
            if isinstance(tensor, dict):
                for obs_k, v in tensor.items():
                    # self.tensors[obs_k] is of shape [N, 1, size] for some reason, hence .unsqueeze(dim=1) of v
                    # for sequences it is [N, 1, seq, size]
                    # note [:] ESSENTIAL for activating lazy tensor
                    # print(v[:][:num_samples].unsqueeze(dim=1).shape)
                    self.tensors[obs_k][self.memory_index : self.memory_index + num_samples].copy_(
                        v[:][:num_samples].unsqueeze(dim=1)
                    )
                    if overflow_samples > 0:
                        # Only copy as many samples as the overflow_samples count
                        overflow_data = v[:][num_samples : num_samples + overflow_samples].unsqueeze(dim=1)
                        # print(f"Overflow data shape: {overflow_data.shape}")
                        # print(f"Target tensor shape: {self.tensors[obs_k][:overflow_samples].shape}")
                        # Ensure we're only copying the exact number of overflow samples
                        self.tensors[obs_k][:overflow_samples].copy_(overflow_data)

                        # print(overflow_samples, num_samples, v[:].shape, self.tensors[obs_k].shape)
                        # print(self.tensors[obs_k][:overflow_samples].shape)
                        # print(v[:][num_samples:].unsqueeze(dim=1).shape)
                        # self.tensors[obs_k][:overflow_samples].copy_(v[:][num_samples:].unsqueeze(dim=1))

            # "actions"
            else:
                # copy the first n samples
                self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                    tensor[:num_samples].unsqueeze(dim=1)
                )
                if overflow_samples > 0:
                    self.tensors[name][:overflow_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))

        # storage remaining samples
        if overflow_samples > 0:
            self.memory_index = overflow_samples
            self.filled = True
        else:
            self.memory_index += num_samples

        # print("length of memory", len(self), "memory index", self.memory_index)

    def add_parallel_samples(self, tensors):
        for name, tensor in tensors.items():
            # "states" and "next_states" are both dicts, but we don't want to overwrite them
            if isinstance(tensor, dict):

                for k in tensor.keys():
                    # k = {policy, aux}
                    if isinstance(tensor[k], dict):
                        # obs_k = {rgb, depth, prop, etc.}
                        for obs_k, v in tensor[k].items():
                            if obs_k in self.tensors:
                                self.tensors[obs_k][self.memory_index].copy_(
                                    v[:]
                                )  # [:] at the end to activate LazyTensors
                    else:
                        # ssl only.......
                        print("HI")
                        # print(self.tensors[k])
                        self.tensors[k][self.memory_index].copy_(tensor[k][:])
            else:
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)

    def sample_all(
        self,
        names: Tuple[str],
        mini_batches: int = 1,
        sequence_length: int = 1,
        augmentations=None,
        random=False,
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        mem_size = (self.memory_size if self.filled else self.memory_index) * self.num_envs
        batch_size = mem_size // mini_batches

        # sample every single guy in memory
        indexes = np.arange(mem_size)
        np.random.shuffle(indexes)
        indexes = indexes.tolist()

        # Split into minibatches
        batches = [indexes[i : i + batch_size] for i in range(0, len(indexes) - batch_size + 1, batch_size)]

        minibatches = []
        for batch in batches:
            minibatch = []
            minibatch_obs_dict = {}

            # this loops through all tensor names. we want to make the observation tensors into a dict though.
            for name in names:
                obs = self.tensors_view[name][batch]

                if name == "rgb" or name == "depth" or name == "gt" or name == "prop" or name == "tactile":
                    minibatch_obs_dict[name] = obs

                # append actions, log_prob, values, returns, advantages
                else:
                    minibatch.append(obs)

            # insert the obs dictionary at the beginning of the list
            minibatch.insert(0, minibatch_obs_dict)

            minibatches.append(minibatch)

        return minibatches

    def augment_obs(self, obs, augmentations):
        # .half() takes up half the space as float()
        if np.argmin(obs.shape[1:]) == 0:
            obs = augmentations(obs.half())
        else:
            obs = augmentations(obs.half().permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        obs = obs.to(torch.uint8)
        return obs
