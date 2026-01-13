"""Memory buffer for storing and sampling RL transitions.

Implements circular buffers for efficient storage of observations, actions, rewards,
and other transition data. Supports both sequential (event-based) and parallel
(environment-based) sampling strategies.
"""

import gym
import gymnasium
import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from multimodal_rl.wrappers.frame_stack import LazyFrames


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
        """Initialize memory buffer with circular buffer storage.
        
        Buffers are torch tensors with shape (memory_size, num_envs, data_size).
        Circular buffers wrap around when full, overwriting oldest data.
        
        Args:
            memory_size: Maximum number of timesteps to store (rollout length).
            num_envs: Number of parallel environments (default: 1).
            device: Device for tensor storage (default: auto-detect CUDA/CPU).
            export: Whether to export memory when filled (default: False).
            export_format: Export format: "pt", "np", or "csv" (default: "pt").
            export_directory: Directory for exports (default: "").
            env_cfg: Environment configuration dictionary (unused, kept for compatibility).
            
        Raises:
            ValueError: If export_format is not supported.
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

        if self.export_format not in ["pt", "np", "csv"]:
            raise ValueError(f"Export format not supported: {self.export_format}")

    def __len__(self) -> int:
        """Return current valid size of memory buffer.
        
        Returns:
            Number of valid samples: memory_size * num_envs if full,
            otherwise memory_index * num_envs.
        """
        return self.memory_size * self.num_envs if self.filled else self.memory_index * self.num_envs

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> torch.Tensor:
        """Retrieve tensor by name.
        
        Args:
            name: Name of tensor to retrieve.
            keepdim: If True, return shape (memory_size, num_envs, size).
                    If False, return flattened shape (memory_size * num_envs, size).
                    
        Returns:
            Requested tensor.
            
        Raises:
            KeyError: If tensor name doesn't exist.
        """
        return self.tensors[name] if keepdim else self.tensors_view[name]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Set tensor values by name.
        
        Args:
            name: Name of tensor to update.
            tensor: New tensor values.
            
        Raises:
            KeyError: If tensor name doesn't exist.
        """
        with torch.no_grad():
            self.tensors[name].copy_(tensor)

    def _get_space_size(
        self, space: Union[int, Tuple[int], gym.Space, gymnasium.Space], keep_dimensions: bool = False
    ) -> Union[Tuple, int]:
        """Compute size (number of elements) of a space or shape.
        
        Supports gym/gymnasium spaces, tuples, lists, and scalars.
        
        Args:
            space: Space or shape specification.
            keep_dimensions: If True, return tuple shape; if False, return total elements.
            
        Returns:
            Size as int (total elements) or tuple (shape).
            
        Raises:
            ValueError: If space type is not supported or keep_dimensions=True with Dict space.
        """
        if isinstance(space, (int, float)):
            return (int(space),) if keep_dimensions else int(space)
        elif isinstance(space, (tuple, list)):
            return tuple(space) if keep_dimensions else np.prod(space)
        elif issubclass(type(space), gym.Space):
            return self._get_gym_space_size(space, keep_dimensions)
        elif issubclass(type(space), gymnasium.Space):
            return self._get_gym_space_size(space, keep_dimensions)
        raise ValueError(f"Unsupported space type: {type(space)}")
    
    def _get_gym_space_size(self, space, keep_dimensions: bool) -> Union[Tuple, int]:
        """Helper to compute size for gym/gymnasium spaces."""
        if isinstance(space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
            return (1,) if keep_dimensions else 1
        elif isinstance(space, (gym.spaces.MultiDiscrete, gymnasium.spaces.MultiDiscrete)):
            return space.nvec.shape[0]
        elif isinstance(space, (gym.spaces.Box, gymnasium.spaces.Box)):
            return tuple(space.shape) if keep_dimensions else np.prod(space.shape)
        elif isinstance(space, (gym.spaces.Dict, gymnasium.spaces.Dict)):
            if keep_dimensions:
                raise ValueError("keep_dimensions=True cannot be used with Dict spaces")
            return sum(self._get_space_size(space.spaces[key], keep_dimensions=False) for key in space.spaces)
        raise ValueError(f"Unsupported gym space type: {type(space)}")

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int], gym.Space, gymnasium.Space],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = True,
    ) -> bool:
        """Create a new tensor in memory buffer.
        
        Tensor shape: (memory_size, num_envs, *size) if keep_dimensions=True,
        otherwise (memory_size, num_envs, total_size).
        
        Args:
            name: Tensor name (must be valid Python identifier).
            size: Size specification (int, tuple, or gym/gymnasium Space).
            dtype: Data type (default: None, uses torch default).
            keep_dimensions: Whether to preserve shape dimensions (default: True).
            
        Returns:
            True if tensor was created, False if it already exists.
            
        Raises:
            ValueError: If tensor exists with different size/dtype.
        """
        size = self._get_space_size(size, keep_dimensions)
        
        # Check if tensor already exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError(
                    f"Tensor '{name}' size mismatch: requested {size}, existing {tensor.size(-1)}"
                )
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Tensor '{name}' dtype mismatch: requested {dtype}, existing {tensor.dtype}"
                )
            return False

        # Create tensor with appropriate shape
        tensor_shape = (
            (self.memory_size, self.num_envs, *size) if keep_dimensions
            else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)

        # Create tensor and store references
        setattr(self, f"_tensor_{name}", torch.zeros(tensor_shape, device=self.device, dtype=dtype))
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        
        # Initialize floating point tensors with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def reset(self) -> None:
        """Reset memory buffer indices and flags.
        
        Old data remains in memory but will be overwritten. Access through
        normal methods may not be guaranteed until buffer is refilled.
        """
        self.filled = False
        self.memory_index = 0

    def add_samples(self, sample_type: str, **tensors: torch.Tensor) -> None:
        """Record samples in memory buffer.
        
        Args:
            sample_type: "sequential" for event-based or "parallel" for environment-based.
            **tensors: Named tensors to store (e.g., states=..., actions=..., rewards=...).
            
        Raises:
            ValueError: If no tensors provided or sample_type is invalid.
        """
        if not tensors:
            raise ValueError("No samples provided. Pass tensors as keyword arguments.")

        if sample_type == "sequential":
            self.add_sequential_samples(tensors)
        elif sample_type == "parallel":
            self.add_parallel_samples(tensors)
            self.memory_index += 1
        else:
            raise ValueError(f"Unsupported sample_type: {sample_type}. Must be 'sequential' or 'parallel'")

        # Wrap around if buffer is full
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

    def add_sequential_samples(self, tensors):
        """Add sequential/event-based samples to memory.
        
        Assumes "prop" observation exists in states dict. Handles overflow
        by wrapping around to beginning of buffer.
        
        Args:
            tensors: Dictionary of named tensors to store.
            
        Raises:
            TypeError: If LazyFrames detected (should be resolved before this).
            ValueError: If observation shape is invalid.
        """
        # Extract observation to determine batch size
        obs = tensors["states"]["prop"]
        
        if isinstance(obs, LazyFrames):
            raise TypeError("LazyFrames must be resolved before adding sequential samples")

        # Determine batch size from observation shape
        if len(obs.shape) == 3:
            num_incoming_samples, seq_length, size = obs.shape
        elif len(obs.shape) == 2:
            num_incoming_samples, size = obs.shape
        else:
            raise ValueError(f"Invalid observation shape: {obs.shape}")

        self.total_samples += num_incoming_samples
        num_samples = min(num_incoming_samples, self.memory_size - self.memory_index)
        overflow_samples = num_incoming_samples - num_samples

        # Store each tensor type
        for name, tensor in tensors.items():
            if isinstance(tensor, dict):
                # Handle nested observation dictionaries
                for obs_k, v in tensor.items():
                    # Copy main samples (unsqueeze for env dimension)
                    self.tensors[obs_k][self.memory_index : self.memory_index + num_samples].copy_(
                        v[:][:num_samples].unsqueeze(dim=1)
                    )
                    # Handle overflow by wrapping to start of buffer
                    if overflow_samples > 0:
                        overflow_data = v[:][num_samples : num_samples + overflow_samples].unsqueeze(dim=1)
                        self.tensors[obs_k][:overflow_samples].copy_(overflow_data)
            else:
                # Handle non-dict tensors (actions, etc.)
                self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                    tensor[:num_samples].unsqueeze(dim=1)
                )
                if overflow_samples > 0:
                    self.tensors[name][:overflow_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))

        # Update memory index
        if overflow_samples > 0:
            self.memory_index = overflow_samples
            self.filled = True
        else:
            self.memory_index += num_samples

    def add_parallel_samples(self, tensors):
        """Add parallel environment samples to memory.
        
        Stores one timestep of data from all parallel environments.
        Handles nested observation dictionaries (policy/aux) and LazyFrames.
        
        Args:
            tensors: Dictionary of named tensors to store.
        """
        for name, tensor in tensors.items():
            if isinstance(tensor, dict):
                # Handle nested observation dictionaries (e.g., {"policy": {...}, "aux": {...}})
                for k in tensor.keys():
                    if isinstance(tensor[k], dict):
                        # Store individual observation types (rgb, depth, prop, etc.)
                        for obs_k, v in tensor[k].items():
                            if obs_k in self.tensors:
                                # [:] activates LazyFrames if present
                                self.tensors[obs_k][self.memory_index].copy_(v[:])
                    else:
                        # Direct tensor storage
                        if k in self.tensors:
                            self.tensors[k][self.memory_index].copy_(tensor[k][:])
            else:
                # Store non-dict tensors (actions, rewards, etc.)
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)

    def sample_all(
        self,
        names: Tuple[str],
        mini_batches: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory and split into mini-batches.
        
        Observations are grouped into a dictionary, while other tensors
        (actions, log_prob, values, etc.) are kept as separate list elements.
        
        Args:
            names: Names of tensors to sample.
            mini_batches: Number of mini-batches to create (default: 1).
            
        Returns:
            List of mini-batches, where each batch is [obs_dict, actions, log_prob, values, returns, advantages].
        """
        mem_size = (self.memory_size if self.filled else self.memory_index) * self.num_envs
        batch_size = mem_size // mini_batches

        # Shuffle all indices and split into batches
        indexes = np.arange(mem_size)
        np.random.shuffle(indexes)
        batches = [
            indexes[i : i + batch_size]
            for i in range(0, len(indexes) - batch_size + 1, batch_size)
        ]

        minibatches = []
        observation_keys = {"rgb", "depth", "gt", "prop", "tactile"}
        
        for batch in batches:
            minibatch = []
            minibatch_obs_dict = {}

            # Group observations into dict, keep other tensors separate
            for name in names:
                obs = self.tensors_view[name][batch]
                if name in observation_keys:
                    minibatch_obs_dict[name] = obs
                else:
                    minibatch.append(obs)

            # Insert observation dict at beginning
            minibatch.insert(0, minibatch_obs_dict)
            minibatches.append(minibatch)

        return minibatches
