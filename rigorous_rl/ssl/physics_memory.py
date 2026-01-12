import numpy as np
import torch
from typing import Tuple, Optional
from rigorous_rl.wrappers.frame_stack import LazyFrames


class DynamicsMemory:
    def __init__(self,
                 env,
                 encoder,
                 value,
                 value_preprocessor,
                 memory_size: int,
                 seq_length: int = 1,                
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Prioritized experience replay buffer for storing state-action transitions.
        
        Args:
            memory_size: Maximum number of transitions to store
            seq_length: Length of each transition sequence
            obs_size: Dimensionality of observations
            action_size: Dimensionality of actions
            alpha: Prioritization exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling correction exponent
            beta_annealing_steps: Steps over which to anneal beta to 1.0
            epsilon: Small constant to add to priorities to ensure non-zero sampling probability
            device: Device to store tensors on
        """
        self.encoder = encoder
        self.value = value 
        self.value_preprocessor = value_preprocessor 

        self.memory_size = memory_size
        self.seq_length = seq_length
        
        self.device = device

        self.env = env
        self.gamma = 0.99
        # Initialize memory buffers
        print(self.env.action_space.shape[0])
        print((memory_size, seq_length, self.env.action_space.shape[0]))

        # to do: try float16 ????
        self.dtype = torch.float32

        
    def reset(self):
        # print("reseting aux memory")
        self.actions = torch.zeros((self.memory_size, self.seq_length, self.env.action_space.shape[0]), device=self.device, dtype=self.dtype)

        observation_names = []
        self.states = {}
        # outer loop of observation space (policy, aux)
        for type_k in sorted(self.env.observation_space.keys()):
            for k, v in self.env.observation_space[type_k].items():
                # Determine the correct dtype based on the observation key
                if k == "tactile":
                    # Check if binary_tactile is enabled in config
                    tactile_cfg = getattr(self.env, "tactile_cfg", None)
                    if tactile_cfg is not None and tactile_cfg.get("binary_tactile", False):
                        storage_dtype = torch.bool
                    else:
                        storage_dtype = self.dtype
                else:
                    storage_dtype = self.dtype
                
                # create next states for the forward_dynamics
                # print(f"AuxiliaryTask: {k}: {type_k} tensor size {v.shape}")
                # forward dynamics use sequence length
                self.states[k] = torch.zeros((self.memory_size, self.seq_length, *v.shape), device=self.device, dtype=storage_dtype)
                # recon we don't
                # self.states[k] = torch.zeros((self.memory_size, *v.shape), device=self.device, dtype=self.dtype)
                observation_names.append(k)
        
        # Initialize tracking variables
        self.memory_index = 0
        self.filled = False
        self.total_samples = 0

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs + env_index`` is returned

        :return: Valid size
        :rtype: int
        """
        return self.memory_size if self.filled else self.memory_index
    
    def add_samples(self, incoming_states, incoming_actions):
        # note this method assumes we will always have "prop" obs
        obs = incoming_states["prop"]

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

        # if we are not filled, add relevant samples sequentially to the memory index
        num_samples = min(num_incoming_samples, self.memory_size - self.memory_index)
        overflow_samples = num_incoming_samples - num_samples
        # print(f"adding {num_samples} seq to a memory index {self.memory_index}. overflow_samples {overflow_samples}") #, remaining_samples)

        # Store transition
        for k, v in incoming_states.items():
            self.states[k][self.memory_index : self.memory_index + num_samples] = v[:][:num_samples]

            if overflow_samples > 0:
                overflow_data = v[:][num_samples:num_samples+overflow_samples]
                self.states[k][:overflow_samples] = overflow_data

        self.actions[self.memory_index : self.memory_index + num_samples] = incoming_actions[:num_samples]

        # don't even bother with the overflow
        if overflow_samples > 0:
            self.actions[:overflow_samples] = incoming_actions[num_samples:]
            self.memory_index = overflow_samples
            self.filled = True
        else:
            self.memory_index += num_samples


    def sample_all(self, mini_batches: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions based on their importance.
        
        Args:
            batch_size: Number of transitions to sample
            update_beta: Whether to update beta parameter for importance sampling
            
        Returns:
            Tuple of (states, actions, next_states, indices, weights)
        """


        mem_size = self.memory_size if self.filled else self.memory_index
        batch_size = mem_size // mini_batches

        # sample every single guy in memory
        indexes = np.arange(mem_size)
        np.random.shuffle(indexes)
               
        indexes = indexes.tolist()

        # Split into minibatches
        batches = [indexes[i:i+batch_size] for i in range(0, len(indexes) - batch_size + 1, batch_size)]

        minibatches = []
        for batch in batches:
            minibatch = []
            minibatch_obs_dict = {}

            for k in self.env.observation_space["policy"].keys():
                minibatch_obs_dict[k] = self.states[k][batch]

            minibatch_actions = self.actions[batch]
            minibatch = [minibatch_obs_dict, minibatch_actions]
            minibatches.append(minibatch)

        return minibatches

    
    def get_buffer_size(self) -> int:
        """Return the current number of transitions stored."""
        return self.memory_size if self.filled else self.memory_index


    def get_obs_dict_list(self, obs_sequence):
        # get individual obs_dict for each state in order to get z
        obs_dict_list = [{} for _ in range(self.seq_length)]
        for k in sorted(obs_sequence.keys()):
            # shape (batch_size, seq_length, input_size)
            obs = obs_sequence[k] 
            for i in range(self.seq_length):
                # saving as (batch_size, input_size)
                obs_dict_list[i][k] = obs[:,i,:]
        return obs_dict_list
    
