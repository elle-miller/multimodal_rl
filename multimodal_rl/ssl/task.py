"""Base class for self-supervised learning auxiliary tasks.

Provides common infrastructure for SSL tasks that can be trained alongside
the main RL objective to improve representation learning.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
from multimodal_rl.rl.memories import Memory


class AuxiliaryTask(ABC):
    """Abstract base class for self-supervised learning auxiliary tasks.
    
    Provides common infrastructure for SSL tasks that improve representation
    learning by predicting future states, reconstructing observations, etc.
    
    Subclasses must implement:
    - set_optimisable_networks(): Return list of networks to optimize
    - create_memory(): Create memory buffer for SSL task
    - sample_minibatches(): Sample batches from memory
    - compute_loss(): Compute SSL loss for a batch
    
    Args:
        aux_task_cfg: Configuration dictionary for the SSL task.
        rl_rollout: Number of RL rollout steps.
        rl_memory: RL memory buffer (may be shared with SSL task).
        encoder: Shared encoder network.
        value: Value network (unused but kept for compatibility).
        value_preprocessor: Value preprocessor (unused but kept for compatibility).
        env: Training environment.
        env_cfg: Environment configuration.
        writer: Writer for logging.
    """

    def __init__(self, aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer):
        # Hyperparameters
        self.aux_loss_weight = aux_task_cfg["loss_weight"]
        self.lr = aux_task_cfg["learning_rate"]
        self.tactile_only = aux_task_cfg.get("tactile_only", False)
        self.seq_length = aux_task_cfg.get("seq_length", 0)
        self.n_rollouts = aux_task_cfg["n_rollouts"]
        self.learning_epochs_ratio = aux_task_cfg.get("learning_epochs_ratio", 1.0)

        # Memory sharing strategy
        if self.seq_length > 0:
            self.use_same_memory = False
        elif self.n_rollouts == 1:
            self.use_same_memory = True
            self.memory = rl_memory
        else:
            raise NotImplementedError("Multiple rollouts with shared memory not implemented")

        self.random_sample = True  # Sample randomly vs sequentially

        # Environment and device setup
        self.rl_rollout = rl_rollout
        self.env = env
        self.env_cfg = env_cfg
        self.num_eval_envs = env_cfg.num_eval_envs
        self.num_training_envs = env.num_envs - self.num_eval_envs
        self.device = env.device
        self.wandb_session = writer.wandb_session
        self.tb_writer = writer.tb_writer

        # Networks (shared with RL)
        self.encoder = encoder
        self.value = value
        self.value_preprocessor = value_preprocessor
        self.z_dim = self.encoder.num_outputs
        self.action_dim = self.env.action_space.shape[0]
        self.augmentations = None

        # Determine which tensors to sample from memory
        self._aux_tensors_names = []
        for type_k in sorted(self.env.observation_space.keys()):
            for k in self.env.observation_space[type_k].keys():
                self._aux_tensors_names.append(k)
        self._aux_tensors_names.append("actions")

        # Target network parameters (for tasks using target networks)
        self.encoder_per_target_update = 1
        self.tau = 0.01

        # Training state
        self.update_step = 0
        self.minibatch_step = 0

        # Mixed precision training
        self._mixed_precision = False
        self._device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)


    def _post_init(self):
        # we need to create networks before this
        self.optimisable_networks = self.set_optimisable_networks()

        self.optimiser = torch.optim.Adam( 
            [param for net in self.optimisable_networks for param in net.parameters()], 
            lr=self.lr,
        )

        if not self.use_same_memory:
            print("*******************AUX USING OWN MEMORY ")
            self.memory = self.create_memory()
            self.memory.reset()
            # self.create_memory_tensors()

    @abstractmethod
    def set_optimisable_networks(self):
        pass

    @abstractmethod
    def create_memory(self):
        pass

    @abstractmethod
    def sample_minibatches(self):
        pass

    @abstractmethod
    def compute_loss(self, minibatch):
        """Compute SSL loss for a minibatch.
        
        Args:
            minibatch: Batch of data from memory.
            
        Returns:
            Tuple of (loss, info_dict) where info_dict contains metrics to log.
        """
        pass

    def create_sequential_memory(self, size=10000):
        """Create memory buffer for sequential/event-based data (e.g., contacts).
        
        Args:
            size: Memory buffer size (default: 10000).
            
        Returns:
            Memory instance configured for sequential storage.
        """
        return Memory(
            memory_size=int(size),
            num_envs=1,
            device=self.device,
            env_cfg=self.env_cfg,
        )
    
    def create_parallel_memory(self):
        """Create memory buffer for parallel environment transitions.
        
        Collects transitions from all training environments for N rollouts.
        
        Returns:
            Memory instance configured for parallel storage.
        """
        return Memory(
            memory_size=self.rl_rollout,
            num_envs=self.num_training_envs,
            device=self.device,
            env_cfg=self.env_cfg,
        )
    
    def create_memory_tensors(self):
        """Create observation and action tensors in auxiliary memory.
        
        Sets up storage for all observation types (RGB, depth, proprioception, etc.)
        with appropriate dtypes.
        """
        for type_k in sorted(self.env.observation_space.keys()):
            for k, v in self.env.observation_space[type_k].items():
                # Determine dtype based on observation type
                storage_dtype = torch.uint8 if k == "rgb" else torch.float32
                self.memory.create_tensor(name=k, size=v.shape, dtype=storage_dtype)
        
        self.memory.create_tensor(name="actions", size=self.env.action_space, dtype=torch.float32)

    def add_samples(self, states, actions=None, next_states=None, terminated=None, truncated=None):
        """Add samples to dedicated auxiliary memory.
        
        Only works when not sharing memory with RL. Subclasses can override
        if they need different sample formats.
        
        Args:
            states: Current state observations.
            actions: Actions taken (optional).
            next_states: Next state observations (optional).
            terminated: Termination flags (optional).
            truncated: Truncation flags (optional).
        """
        if not self.use_same_memory:
            self.memory.add_samples(
                type="parallel",
                states=states,
                actions=actions,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated
            )

    def soft_update_params(self, net, target_net, tau=0.05):
        """Soft update target network parameters using exponential moving average.
        
        Updates target network: target = tau * source + (1 - tau) * target
        Only updates every encoder_per_target_update steps.
        
        Args:
            net: Source network.
            target_net: Target network to update.
            tau: Update coefficient (default: 0.05).
        """
        if (self.update_step % self.encoder_per_target_update) == 0:
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def separate_memory_tensors(self, sampled_states):
        """Separate policy and auxiliary observation tensors.
        
        Args:
            sampled_states: Dictionary of sampled states.
            
        Returns:
            Tuple of (policy_states_dict, aux_states_dict).
        """
        sampled_states_dict = {}
        for k in sorted(self.env.observation_space.keys()):
            sampled_states_dict[k] = {}
            for obs_k in sorted(self.env.observation_space[k].keys()):
                sampled_states_dict[k][obs_k] = sampled_states[obs_k]
        return sampled_states_dict, None
    
    def set_networks_mode(self, networks, mode='train'):
        for net in networks:
            if mode == 'train':
                net.train()
            else:
                net.eval()

    def evaluate_binary_predictions(self, predictions, targets, step=0, threshold=0.5):
        
        # Convert to binary predictions using threshold
        binary_preds = (predictions >= threshold).float()
        
        # Count total correct predictions
        correct = (binary_preds == targets).float().sum().item()
        total = targets.numel()
        
        # Count correct 1s and 0s separately
        true_positives = ((binary_preds == 1) & (targets == 1)).float().sum().item()
        false_positives = ((binary_preds == 1) & (targets == 0)).float().sum().item()
        true_negatives = ((binary_preds == 0) & (targets == 0)).float().sum().item()
        false_negatives = ((binary_preds == 0) & (targets == 1)).float().sum().item()

        # Count total 1s and 0s
        total_positives = (targets == 1).float().sum().item()
        total_negatives = (targets == 0).float().sum().item()
        
        # Calculate metrics
        accuracy = correct / total
        precision = true_positives / max(1, (binary_preds == 1).float().sum().item())
        recall = true_positives / max(1, total_positives)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        self.minibatch_step += 1

        tp_rate = true_positives / max(1, total_positives)
        fp_rate = false_positives / max(1, total_positives)
        tn_rate = true_negatives / max(1, total_negatives)
        fn_rate = false_negatives / max(1, total_negatives)

        return {
            f'Tactile / accuracy @t={step}': accuracy,
            f'Tactile / precision @t={step}': precision,
            f'Tactile / recall @t={step}': recall,
            f'Tactile / f1 @t={step}': f1,
            
            f'Tactile / true_positive_rate @t={step}': tp_rate,
            f'Tactile / true_negative_rate @t={step}': tn_rate,
            f'Tactile / false_positive_rate @t={step}': fp_rate,
            f'Tactile / false_negative_rate @t={step}': fn_rate,

            f'Tactile / tp': true_positives,
            f'Tactile / tn': true_negatives,
            f'Tactile / fp': false_positives,
            f'Tactile / fn': false_negatives,
            # f'Tactile / correct @t={step}': correct,
            # f'Tactile / total': total
        }


