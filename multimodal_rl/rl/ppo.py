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
from multimodal_rl.rl.pcgrad import PCGrad
from multimodal_rl.rl.memories import Memory
from multimodal_rl.rl.kl_adaptive_scheduler import KLAdaptiveLR

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
    "multi_critic": None,  # Dict with keys: num_critics, gammas, weights, names. If None, single critic.
    "state_dependent_log_std": False,  # If True, log_std is computed from state; if False, uses learnable parameter
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
        value: Value network (DeterministicValue) or MultiCritic wrapper for multiple critics.
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
        self.memory = memory
        self.global_step = 0
        self.num_train_envs = self.memory.num_envs if self.memory is not None else 0

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
        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # Multi-critic configuration
        multi_critic_cfg = self.cfg.get("multi_critic", None)
        if multi_critic_cfg:
            self._num_critics = multi_critic_cfg["num_critics"]
            self._gammas = multi_critic_cfg["gammas"]
            self._critic_weights = torch.tensor(multi_critic_cfg["weights"], device=self.device, dtype=self.dtype)
            self._critic_names = multi_critic_cfg["names"]
            # Validate lengths
            assert len(self._gammas) == self._num_critics, \
                f"gammas length ({len(self._gammas)}) must match num_critics ({self._num_critics})"
            assert len(self._critic_weights) == self._num_critics, \
                f"weights length ({len(multi_critic_cfg['weights'])}) must match num_critics ({self._num_critics})"
            assert len(self._critic_names) == self._num_critics, \
                f"names length ({len(self._critic_names)}) must match num_critics ({self._num_critics})"
        else:
            self._num_critics = 1
            self._gammas = [self._discount_factor]
            self._critic_weights = torch.ones(1, device=self.device, dtype=self.dtype)
            self._critic_names = ["default"]

        # Check if policy uses state-dependent log_std (from config or by inspecting policy)
        self._state_dependent_log_std = self.cfg.get("state_dependent_log_std", False)

        # Networks
        self.policy = policy
        self.value = value
        self.encoder = encoder
        self.policy.eval()
        self.value.eval()
        self.encoder.eval()

        # Separate optimizers for each network component
        adam_epsilon = 1e-5
        self.policy_optimiser = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate, eps=adam_epsilon)
        self.value_optimiser = torch.optim.Adam(self.value.parameters(), lr=self._learning_rate, eps=adam_epsilon)
        self.encoder_optimiser = torch.optim.Adam(self.encoder.parameters(), lr=self._learning_rate, eps=adam_epsilon)

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

            if self._num_critics > 1:
                for i, critic in enumerate(self.value.critics):
                    if critic.value_preprocessor is not None:
                        self.writer.checkpoint_modules[f"value_preprocessor_{i}"] = critic.value_preprocessor
            else:
                self.writer.checkpoint_modules["value_preprocessor"] = self.value.value_preprocessor
            
        self.num_actions = self.action_space.shape[0]

        # Learning rate schedulers (if specified)
        # Can create separate schedulers for each optimizer or use the same scheduler for all
        self._learning_rate_schedulers = {}
        scheduler_kwargs = self.cfg.get("learning_rate_scheduler_kwargs", {})
        
        if self._learning_rate_scheduler is not None:
            scheduler_name_or_class = self._learning_rate_scheduler
            
            # Check if it's KLAdaptiveLR (string or class)
            is_kl_adaptive = (
                scheduler_name_or_class == "KLAdaptiveLR" or
                (isinstance(scheduler_name_or_class, type) and issubclass(scheduler_name_or_class, KLAdaptiveLR))
            )
            policy_optim = self.policy_optimiser
            
            if is_kl_adaptive:
                # KLAdaptiveLR needs KL divergence, typically only applied to policy
                # Filter out apply_to_all from kwargs before passing to KLAdaptiveLR
                filtered_kwargs = {k: v for k, v in scheduler_kwargs.items() if k != "apply_to_all"}
                self._learning_rate_schedulers["policy"] = KLAdaptiveLR(
                    policy_optim,
                    **filtered_kwargs
                )
                # Optionally apply same learning rate to other optimizers
                if scheduler_kwargs.get("apply_to_all", False):
                    self._learning_rate_schedulers["value"] = KLAdaptiveLR(
                        self.value_optimiser,
                        **filtered_kwargs
                    )
                    self._learning_rate_schedulers["encoder"] = KLAdaptiveLR(
                        self.encoder_optimiser,
                        **filtered_kwargs
                    )
            else:
                # Standard PyTorch scheduler - apply to all optimizers
                scheduler_class = scheduler_name_or_class
                if isinstance(scheduler_class, str):
                    # Import from torch.optim.lr_scheduler
                    import torch.optim.lr_scheduler as lr_scheduler
                    scheduler_class = getattr(lr_scheduler, scheduler_class)
                
                self._learning_rate_schedulers["policy"] = scheduler_class(
                    policy_optim,
                    **scheduler_kwargs
                )
                self._learning_rate_schedulers["value"] = scheduler_class(
                    self.value_optimiser,
                    **scheduler_kwargs
                )
                self._learning_rate_schedulers["encoder"] = scheduler_class(
                    self.encoder_optimiser,
                    **scheduler_kwargs
                )

        self.update_step = 0
        self.epoch_step = 0

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
            self.memory.create_tensor(name="rewards", size=self._num_critics, dtype=self.dtype)
            self.memory.create_tensor(name="values", size=self._num_critics, dtype=self.dtype)
            self.memory.create_tensor(name="returns", size=self._num_critics, dtype=self.dtype)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=self.dtype)
            self.memory.create_tensor(name="advantages", size=self._num_critics, dtype=self.dtype)

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
            rewards: Rewards received. For multi-critic, shape should be (num_envs, num_critics).
                     For single critic, shape should be (num_envs, 1) or (num_envs,).
            next_states: Next state observations.
            terminated: Episode termination flags.
            truncated: Episode truncation flags.
            timestep: Current timestep (unused, kept for compatibility).
        """
        if self.memory is None:
            return

        self._current_next_states = next_states

        # Ensure rewards have correct shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        
        # check
        if self._num_critics != rewards.shape[1]:
            raise ValueError(
                f"Expected rewards shape (num_envs, {self._num_critics}) for multi-critic, "
                f"got {rewards.shape}"
            )

        # Compute value estimates (no gradients needed during rollout collection)
        with torch.no_grad():
            z = self.encoder(states)
            values = self.value.compute_value(z, inverse=True)

        # Time-limit bootstrapping: add discounted value to reward at truncation
        if self._time_limit_bootstrap:
            if self._num_critics > 1:
                gammas = torch.tensor(self._gammas, device=self.device, dtype=self.dtype)
                rewards += gammas * values * truncated
            else:
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

    def compute_gae(self, rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            gamma: float = 0.99,
            lambda_coefficient: float = 0.95,
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
                + gamma * not_dones[i] * (next_value + lambda_coefficient * advantage)
            )
            advantages[i] = advantage
            
        # Compute returns and normalize advantages
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


    def _update(self) -> bool:
        """Execute PPO update step.
        
        Computes GAE advantages, samples mini-batches, and performs
        multiple learning epochs with gradient updates.
        
        Returns:
            True if NaN/Inf detected (should prune trial), False otherwise.
        """


        # Compute value estimates for terminal states: size (num_envs, num_critics)
        with torch.no_grad():
            self.value.eval()
            z = self.encoder(self._current_next_states)
            last_values = self.value.compute_value(z, inverse=True)          
            self.value.train()

        # Get stored values and compute GAE - size (rollout, num_envs, num_critics)
        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        dones = (
            self.memory.get_tensor_by_name("terminated")
            | self.memory.get_tensor_by_name("truncated")
        )

        if self._num_critics > 1:
            all_returns = []
            all_advantages = []
            for i in range(self._num_critics):
                rewards_i = rewards[:, :, i].unsqueeze(-1)
                values_i = values[:, :, i].unsqueeze(-1)
                last_values_i = last_values[:, i].unsqueeze(-1)
                critic_returns, critic_advantages = self.compute_gae(
                    rewards=rewards_i,
                    dones=dones,
                    values=values_i,
                    last_values=last_values_i,
                    gamma=self._gammas[i],
                    lambda_coefficient=self._lambda,
                )
                all_returns.append(critic_returns)
                all_advantages.append(critic_advantages)
            
            returns = torch.cat(all_returns, dim=-1) # size: (rollout, num_envs, num_critics)
            advantages = torch.cat(all_advantages, dim=-1) # size: (rollout, num_envs, num_critics)

        else:
            returns, advantages = self.compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                last_values=last_values,
                gamma=self._discount_factor,
                lambda_coefficient=self._lambda,
            )

        # Preprocess values and returns for training
        processed_values = self.value.value_preprocessor(values, train=True)
        processed_returns = self.value.value_preprocessor(returns, train=True)
        self.memory.set_tensor_by_name("values", processed_values)
        self.memory.set_tensor_by_name("returns", processed_returns)
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

                sampled_states = {"policy": sampled_states}

                # Retain gradients for log_std parameter (only if not state-dependent)
                if not self._state_dependent_log_std and hasattr(self.policy, 'log_std_parameter') and self.policy.log_std_parameter.requires_grad:
                    self.policy.log_std_parameter.retain_grad()

                # Policy path: compute new action probabilities
                # train only for first epoch
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

                # Recompute ratio WITH gradients for policy optimization
                ratio = torch.exp(next_log_prob - sampled_log_prob)

                # Compute per-critic clipped policy losses
                if self._num_critics > 1:
                    advantages = (sampled_advantages * self._critic_weights).sum(dim=-1, keepdim=True)
                else:
                    advantages = sampled_advantages
                surr = advantages * ratio
                surr_clipped = advantages * torch.clamp(
                    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                )
                policy_loss = -torch.min(surr, surr_clipped).mean() 

                # Value path: compute value predictions
                z_value = self.encoder(sampled_states)
                z_value.requires_grad_(True)
                predicted_values = self.value.compute_value(z_value)
                if self._value_clip > 0:
                    predicted_values = sampled_values + torch.clamp(
                        predicted_values - sampled_values,
                        min=-self._value_clip,
                        max=self._value_clip,
                    )

                if self._num_critics > 1:
                    value_losses = []
                    for i in range(self._num_critics):
                        predicted_values_i = predicted_values[:, i]
                        sampled_returns_i = sampled_returns[:, i]
                        value_loss = self._value_loss_scale * F.mse_loss(sampled_returns_i, predicted_values_i)
                        value_losses.append(value_loss)
                    value_loss = sum(value_losses) / self._num_critics
                else:
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
                # if torch.isnan(loss).any() or torch.isinf(loss).any():
                #     print(f"NaN/Inf detected in loss at epoch {epoch}, minibatch {i}")
                #     self._check_instability(policy_loss, "policy_loss")
                #     self._check_instability(value_loss, "value_loss")
                #     self._check_instability(predicted_values, "predicted_values")
                #     self._check_instability(sampled_actions, "sampled_actions")
                #     self._check_instability(sampled_states["policy"].get("prop"), "prop")
                #     self._check_instability(sampled_values, "sampled_values")
                #     self._check_instability(sampled_returns, "sampled_returns")
                #     self._check_instability(sampled_log_prob, "sampled_log_prob")
                #     self._check_instability(sampled_advantages, "sampled_advantages")
                #     if self.wandb_session is not None:
                #         self.wandb_session.finish()
                #     return True

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(),
                            self.value.parameters(),
                            self.encoder.parameters()
                        ),
                        self._grad_norm_clip,
                    )

                # Update parameters
                self.encoder_optimiser.step()
                self.policy_optimiser.step()
                self.value_optimiser.step()
                if self.ssl_task is not None:
                    self.ssl_task.optimiser.step()

                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

                if self.ssl_task is not None and epoch < aux_learning_epochs:
                    cumulative_aux_loss += aux_loss.item()

                if self._entropy_loss_scale > 0:
                    cumulative_entropy_loss += entropy_loss.item()

            # Update learning rate schedulers if specified
            if self._learning_rate_schedulers:
                avg_kl = torch.tensor(kl_divergences, device=self.device).mean() if kl_divergences else None
                
                for name, scheduler in self._learning_rate_schedulers.items():
                    if isinstance(scheduler, KLAdaptiveLR):
                        if avg_kl is not None:
                            scheduler.step(avg_kl.item())
                    else:
                        # Standard PyTorch scheduler
                        scheduler.step()

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
                self.tb_writer.add_scalar("learning_rate/policy", self.policy_optimiser.param_groups[0]["lr"], global_step=self.update_step)
                self.tb_writer.add_scalar("learning_rate/value", self.value_optimiser.param_groups[0]["lr"], global_step=self.update_step)
                self.tb_writer.add_scalar("learning_rate/encoder", self.encoder_optimiser.param_groups[0]["lr"], global_step=self.update_step)

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
            if self._num_critics > 1:
                # Per-critic advantage stats
                for i, name in enumerate(self._critic_names):
                    adv_slice = sampled_advantages[:, i] if sampled_advantages.dim() > 1 and sampled_advantages.shape[-1] > 1 else sampled_advantages.squeeze(-1)
                    wandb_dict[f"Advantages / {name}_mean"] = adv_slice.mean().item()
                    wandb_dict[f"Advantages / {name}_std"] = adv_slice.std().item()

                # Total advantage stats (weighted sum)
                total_adv = (sampled_advantages * self._critic_weights).sum(dim=-1) if sampled_advantages.dim() > 1 and sampled_advantages.shape[-1] > 1 else sampled_advantages.squeeze(-1)
                wandb_dict["Advantages / total_mean"] = total_adv.mean().item()
                wandb_dict["Advantages / total_std"] = total_adv.std().item()

                # Per-critic value preprocessor stats
                for critic_idx, critic in enumerate(self.value.critics):
                    name = self._critic_names[critic_idx]
                    wandb_dict[f"Value scaler / {name} mu_mean"] = critic.value_preprocessor.running_mean_mean
                    wandb_dict[f"Value scaler / {name} variance_mean"] = critic.value_preprocessor.running_variance_mean
            else:
                wandb_dict.update({
                    "Value scaler / mu_mean": self.value.value_preprocessor.running_mean_mean,
                    "Value scaler / variance_mean": self.value.value_preprocessor.running_variance_mean,
                })

            # Log policy standard deviation
            wandb_dict["Policy / Standard deviation"] = self.policy.distribution().stddev.mean().item()
            
            # Log learning rates for each optimizer
            wandb_dict["Learning Rate / Policy"] = self.policy_optimiser.param_groups[0]["lr"]
            wandb_dict["Learning Rate / Value"] = self.value_optimiser.param_groups[0]["lr"]
            wandb_dict["Learning Rate / Encoder"] = self.encoder_optimiser.param_groups[0]["lr"]
            
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

