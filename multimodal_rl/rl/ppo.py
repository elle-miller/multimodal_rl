"""Proximal Policy Optimization (PPO) implementation.

Clean, readable PPO implementation supporting multimodal observations,
self-supervised learning auxiliary tasks, and separate optimizers for
encoder, policy, and value networks.

Reference: https://arxiv.org/abs/1707.06347
"""

import copy
import itertools

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
from multimodal_rl.rl.memories import Memory

PPO_DEFAULT_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 1.0,
    "kl_threshold": 0,
    "time_limit_bootstrap": False,
    "experiment": {
        "directory": "",
        "experiment_name": "",
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}


class PPO:
    """Proximal Policy Optimization agent.
    
    Implements PPO with support for:
    - Multimodal observations (RGB, depth, proprioception, tactile, ground-truth)
    - Self-supervised learning auxiliary tasks
    - Separate optimizers for encoder, policy, and value networks
    - Generalized Advantage Estimation (GAE)
    - Gradient clipping and value clipping for stability
    
    Args:
        encoder: Encoder network for multimodal observations.
        policy: Policy network (GaussianPolicy).
        value: Value network (DeterministicValue).
        value_preprocessor: Value preprocessor (e.g., RunningStandardScaler).
        memory: Memory buffer for storing transitions (optional).
        observation_space: Observation space specification.
        action_space: Action space specification.
        device: Device for computation (default: auto-detect).
        cfg: Configuration dictionary overriding defaults.
        ssl_task: Optional SSL auxiliary task.
        writer: Writer instance for logging.
        dtype: Data type for tensors (default: float32).
        debug: Enable debug checks for NaN/Inf (default: False).
    """

    def __init__(
        self,
        encoder,
        policy,
        value,
        value_preprocessor,
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        ssl_task=None,
        writer=None,
        dtype=torch.float32,
        debug: bool = False
    ) -> None:
        # Merge config with defaults
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg if cfg is not None else {}
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )
        self.dtype = dtype
        self._device_type = torch.device(self.device).type
        self.memory = memory
        self.global_step = 0
        self.num_train_envs = self.memory.num_envs

        # Writer setup
        self.writer = writer
        if self.writer is None:
            self.wandb_session = None
            self.tb_writer = None
        else:
            self.wandb_session = writer.wandb_session
            self.tb_writer = writer.tb_writer
        self.ssl_task = ssl_task

        # Hyperparameters
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0
        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._kl_threshold = self.cfg["kl_threshold"]
        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self._value_preprocessor = value_preprocessor
        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # Networks
        self.policy = policy
        self.value = value
        self.encoder = encoder
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        # Separate optimizers for each network component
        self.policy_optimiser = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        self.value_optimiser = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate)
        self.encoder_optimiser = torch.optim.Adam(self.encoder.parameters(), lr=self._learning_rate)

        # Register modules for checkpointing
        if self.writer is not None:
            self.writer.checkpoint_modules["policy"] = self.policy
            self.writer.checkpoint_modules["value"] = self.value
            self.writer.checkpoint_modules["encoder"] = self.encoder
            self.writer.checkpoint_modules["policy_optimiser"] = self.policy_optimiser
            self.writer.checkpoint_modules["value_optimiser"] = self.value_optimiser
            self.writer.checkpoint_modules["encoder_optimiser"] = self.encoder_optimiser

            if self.encoder.state_preprocessor is not None:
                self.writer.checkpoint_modules["state_preprocessor"] = self.encoder.state_preprocessor

        if self._value_preprocessor is not None:
            self.writer.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor
            
        self.num_actions = self.action_space.shape[0]

        # Learning rate scheduler (if specified)
        if self._learning_rate_scheduler is not None:
            # Note: scheduler would need a combined optimizer, currently not implemented
            pass

        self.update_step = 0
        self.epoch_step = 0

        # Mixed precision training (disabled due to numerical instabilities)
        self._mixed_precision = False
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)

        # Initialize memory tensors
        if self.memory is not None:
            self.observation_names = []
            
            # Create observation tensors with appropriate dtypes
            for k, v in self.observation_space["policy"].items():
                if k == "rgb":
                    storage_dtype = torch.uint8
                elif k == "depth":
                    storage_dtype = torch.float32  # High precision for depth measurements
                elif k == "tactile":
                    # Check if binary tactile encoding is enabled
                    tactile_cfg = self.cfg.get("observations", {}).get("tactile_cfg", {})
                    storage_dtype = torch.bool if tactile_cfg.get("binary_tactile", False) else torch.float32
                else:
                    storage_dtype = self.dtype
                
                self.memory.create_tensor(name=k, size=v.shape, dtype=storage_dtype)
                self.observation_names.append(k)

            # Create transition tensors
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=self.dtype)
            self.memory.create_tensor(name="rewards", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="values", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="returns", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="advantages", size=1, dtype=self.dtype)

            self._tensors_names = self.observation_names + [
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
            ]

        self._current_next_states = None

    def record_transition(
        self,
        states: Union[torch.Tensor, Dict],
        actions: torch.Tensor,
        log_prob: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        timestep: int,
    ) -> None:
        """Record an environment transition in memory.
        
        Computes value estimates, handles time-limit bootstrapping,
        and stores the transition for later PPO updates.
        
        Args:
            states: Current state observations.
            actions: Actions taken by the agent.
            log_prob: Log probabilities of taken actions.
            rewards: Rewards received.
            next_states: Next state observations.
            terminated: Episode termination flags.
            truncated: Episode truncation flags.
            timestep: Current timestep (unused, kept for compatibility).
        """
        if self.memory is None:
            return

        self._current_next_states = next_states

        # Compute value estimates
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            z = self.encoder(states)
            values = self.value.compute_value(z)
            
            # Inverse preprocess to get raw value estimates
            values = self._value_preprocessor(values, inverse=True)
            

        # Time-limit bootstrapping: add discounted value to reward at truncation
        if self._time_limit_bootstrap:
            rewards += self._discount_factor * values * truncated

        # Store transition in memory
        self.memory.add_samples(
            sample_type="parallel",
            states={"policy": states["policy"]},
            actions=actions,
            rewards=rewards,
            next_states=None,
            terminated=terminated,
            truncated=truncated,
            log_prob=log_prob,
            values=values,
        )

    def _update(self) -> bool:
        """Execute PPO update step.
        
        Computes GAE advantages, samples mini-batches, and performs
        multiple learning epochs with gradient updates.
        
        Returns:
            True if NaN/Inf detected (should prune trial), False otherwise.
        """
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            discount_factor: float,
            lambda_coefficient: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute Generalized Advantage Estimation (GAE).
            
            Args:
                rewards: Rewards tensor.
                dones: Done flags (terminated or truncated).
                values: Value estimates.
                last_values: Value estimates for terminal states.
                discount_factor: Discount factor (gamma).
                lambda_coefficient: GAE lambda parameter.
                
            Returns:
                Tuple of (returns, advantages).
            """
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            advantage = 0.0

            # Compute advantages backwards through time
            for i in reversed(range(memory_size)):
                next_value = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_value + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            
            # Compute returns and normalize advantages
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # Compute value estimates for terminal states
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.eval()
            z = self.encoder(self._current_next_states)
            last_values = self.value.compute_value(z)
            last_values = self._value_preprocessor(last_values, inverse=True)
            self.value.train()

        # Get stored values and compute GAE
        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        dones = (
            self.memory.get_tensor_by_name("terminated")
            | self.memory.get_tensor_by_name("truncated")
        )

        returns, advantages = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        # Preprocess values and returns for training
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        # Sample SSL batches if using separate memory
        if self.ssl_task is not None:
            if self.ssl_task.use_same_memory:
                sampled_aux_batches = sampled_batches
            else:
                sampled_aux_batches = self.ssl_task.memory.sample_all(mini_batches=self._mini_batches)
            assert len(sampled_aux_batches) == len(sampled_batches)
        else:
            sampled_aux_batches = None

        # Set networks to training mode
        self.policy.train()
        self.value.train()
        self.encoder.train()

        # Initialize loss accumulators
        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_aux_loss = 0.0

        # Determine SSL learning epochs
        aux_learning_epochs = (
            max(1, int(self._learning_epochs * self.ssl_task.learning_epochs_ratio))
            if self.ssl_task is not None else 1
        )

        # Learning epochs loop
        for epoch in range(self._learning_epochs):
            kl_divergences = []
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0

            # Mini-batches loop
            for i, minibatch in enumerate(sampled_batches):
                if len(minibatch) != 6:
                    raise ValueError(f"Expected 6 elements in minibatch, got {len(minibatch)}")
                
                (
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = minibatch

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = {"policy": sampled_states}

                    # Retain gradients for log_std parameter
                    self.policy.log_std_parameter.retain_grad()

                    # Policy path: compute new action probabilities
                    z_policy = self.encoder(sampled_states, train=not epoch)
                    z_policy.requires_grad_(True)
                    _, next_log_prob, _ = self.policy.act(z_policy, taken_actions=sampled_actions)
                
                    # Compute approximate KL divergence for early stopping
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # Early stopping if KL divergence exceeds threshold
                    if self._kl_threshold > 0 and kl_divergence > self._kl_threshold:
                        break

                    # Compute entropy loss
                    entropy_loss = (
                        -self._entropy_loss_scale * self.policy.get_entropy().mean()
                        if self._entropy_loss_scale > 0
                        else torch.tensor(0.0, device=self.device)
                    )

                    # Compute clipped policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clamp(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Value path: compute value predictions
                    z_value = self.encoder(sampled_states)
                    z_value.requires_grad_(True)
                    predicted_values = self.value.compute_value(z_value)

                    # Clip value updates for stability
                    if self._value_clip > 0:
                        predicted_values = sampled_values + torch.clamp(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )

                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # Compute SSL auxiliary loss if enabled
                    if self.ssl_task is not None and epoch < aux_learning_epochs:
                        aux_minibatch_full = sampled_aux_batches[i]
                        aux_minibatch = (aux_minibatch_full[0], aux_minibatch_full[1])
                        aux_loss, aux_info = self.ssl_task.compute_loss(aux_minibatch)
                        aux_loss *= self.ssl_task.aux_loss_weight
                        loss = policy_loss + entropy_loss + value_loss + aux_loss
                    else:
                        loss = policy_loss + entropy_loss + value_loss
                        aux_info = {}

                # Optimization step
                self.encoder_optimiser.zero_grad()
                self.policy_optimiser.zero_grad()
                self.value_optimiser.zero_grad()
                if self.ssl_task is not None:
                    self.ssl_task.optimiser.zero_grad()

                # Check for numerical instability
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN/Inf detected in loss at epoch {epoch}, minibatch {i}")
                    self._check_instability(policy_loss, "policy_loss")
                    self._check_instability(value_loss, "value_loss")
                    self._check_instability(predicted_values, "predicted_values")
                    self._check_instability(sampled_actions, "sampled_actions")
                    self._check_instability(sampled_states["policy"].get("prop"), "prop")
                    self._check_instability(sampled_values, "sampled_values")
                    self._check_instability(sampled_returns, "sampled_returns")
                    self._check_instability(sampled_log_prob, "sampled_log_prob")
                    self._check_instability(sampled_advantages, "sampled_advantages")
                    if self.wandb_session is not None:
                        self.wandb_session.finish()
                    return True

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.encoder_optimiser)
                    self.scaler.unscale_(self.policy_optimiser)
                    self.scaler.unscale_(self.value_optimiser)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(),
                            self.value.parameters(),
                            self.encoder.parameters()
                        ),
                        self._grad_norm_clip,
                    )

                # Update parameters
                self.scaler.step(self.encoder_optimiser)
                self.scaler.step(self.policy_optimiser)
                self.scaler.step(self.value_optimiser)
                if self.ssl_task is not None:
                    self.scaler.step(self.ssl_task.optimiser)
                self.scaler.update()

                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

                if self.ssl_task is not None and epoch < aux_learning_epochs:
                    cumulative_aux_loss += aux_loss.item()

                if self._entropy_loss_scale > 0:
                    cumulative_entropy_loss += entropy_loss.item()

            # Update learning rate scheduler if specified
            if self._learning_rate_scheduler is not None:
                # Note: scheduler would need combined optimizer
                pass

            self.epoch_step += 1
            cumulative_policy_loss += epoch_policy_loss
            cumulative_value_loss += epoch_value_loss


        # Log metrics to wandb and tensorboard
        if self.wandb_session is not None:
            num_updates = self._learning_epochs * self._mini_batches
            wandb_dict = {
                "global_step": self.update_step,
                "Loss / Policy loss": cumulative_policy_loss / num_updates,
                "Loss / Value loss": cumulative_value_loss / num_updates,
            }

            # Log to TensorBoard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("policy_loss", wandb_dict["Loss / Policy loss"], global_step=self.global_step)
                self.tb_writer.add_scalar("value_loss", wandb_dict["Loss / Value loss"], global_step=self.global_step)

            # Log SSL task metrics
            if self.ssl_task is not None:
                avg_aux_loss = cumulative_aux_loss / num_updates
                wandb_dict["Loss / Aux loss"] = avg_aux_loss
                wandb_dict["Loss / Entropy loss"] = cumulative_entropy_loss / num_updates
                wandb_dict["Memory / size"] = len(self.memory)
                wandb_dict["Memory / memory_index"] = self.ssl_task.memory.memory_index
                wandb_dict["Memory / N_filled"] = int(
                    self.ssl_task.memory.total_samples / self.ssl_task.memory.memory_size
                )
                wandb_dict["Memory / filled"] = float(self.ssl_task.memory.filled)

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("aux_loss", avg_aux_loss, global_step=self.update_step)
                
                # Log SSL-specific metrics
                for k, v in aux_info.items():
                    wandb_dict[k] = v
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(k, v, global_step=self.update_step)

            # Log value preprocessor statistics
            if isinstance(self._value_preprocessor, RunningStandardScaler):
                wandb_dict.update({
                    "Value scaler / running_mean_mean": self._value_preprocessor.running_mean_mean,
                    "Value scaler / running_mean_median": self._value_preprocessor.running_mean_median,
                    "Value scaler / running_mean_min": self._value_preprocessor.running_mean_min,
                    "Value scaler / running_mean_max": self._value_preprocessor.running_mean_max,
                    "Value scaler / running_variance_mean": self._value_preprocessor.running_variance_mean,
                    "Value scaler / running_variance_median": self._value_preprocessor.running_variance_median,
                    "Value scaler / running_variance_min": self._value_preprocessor.running_variance_min,
                    "Value scaler / running_variance_max": self._value_preprocessor.running_variance_max,
                })

            # Log policy standard deviation
            wandb_dict["Policy / Standard deviation"] = self.policy.distribution().stddev.mean().item()
            self.wandb_session.log(wandb_dict)

        # Update step counters
        self.update_step += 1
        self.global_step = self.update_step * self._rollouts * self.num_train_envs

        # Set networks to eval mode for rollout collection
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        return False

    def _check_instability(self, x, name):
        """Check for NaN/Inf values in tensor (debug mode only).
        
        Args:
            x: Tensor to check.
            name: Name for logging.
        """
        if x is None:
            return
        if torch.isnan(x).any():
            print(f"PPO / {name} contains NaN")
        if torch.isinf(x).any():
            print(f"PPO / {name} contains Inf")

    def load(self, path: str) -> None:
        """Load model checkpoints from file.
        
        Loads state dictionaries for all registered checkpoint modules.
        
        Args:
            path: Path to checkpoint file.
        """
        if self.writer is None:
            raise ValueError("Cannot load checkpoint: writer is None")
            
        modules = torch.load(path, map_location=self.device)
        if isinstance(modules, dict):
            for name, data in modules.items():
                module = self.writer.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError(f"Module {name} does not support load_state_dict")
                else:
                    print(f"Warning: Cannot load '{name}' module (not registered)")

    def _empty_preprocessor(self, _input, *args, **kwargs):
        """Identity preprocessor (no-op).
        
        Defined as a method instead of lambda for PyTorch multiprocessing compatibility.
        
        Args:
            _input: Input tensor.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.
            
        Returns:
            Input tensor unchanged.
        """
        return _input