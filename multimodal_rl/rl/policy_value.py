"""Policy and value network implementations for PPO.

Provides Gaussian policy (stochastic) and deterministic value function networks.
"""

import itertools

import gym
import gymnasium
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any, List, Mapping, Optional, Tuple, Union
from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
import numpy as np

from multimodal_rl.models.mlp import MLP

_ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}

def init_ppo_weights(module, std=np.sqrt(2), bias_const=0.0):
    """
    Standard PPO initialization for Linear layers.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=std)
        nn.init.constant_(module.bias, bias_const)

class GaussianPolicy(torch.nn.Module):
    """Gaussian policy network for continuous action spaces.
    
    Outputs mean actions via MLP and learns a learnable log standard deviation.
    Uses reparameterization trick for sampling and supports action clipping.
    
    Args:
        z_dim: Dimension of input latent representation.
        observation_space: Observation space (unused, kept for API compatibility).
        action_space: Action space defining action bounds.
        device: Device to run computations on.
        initial_log_std: Initial value for log standard deviation (default: 0).
        clip_actions: Whether to clip actions to action space bounds (default: False).
        clip_log_std: Whether to clip log standard deviation (default: True).
        min_log_std: Minimum log standard deviation if clipping enabled (default: -20).
        max_log_std: Maximum log standard deviation if clipping enabled (default: 2).
        hiddens: List of hidden layer sizes (default: [256, 128, 64]).
        activations: List of activation function names (default: ["elu", "elu", "elu", "tanh"]).
        reduction: How to reduce log probability across action dimensions:
            "sum", "mean", "prod", or "none" (default: "sum").
        state_dependent_log_std: If True, log_std is computed from state via a head;
            if False, uses a learnable parameter (default: False).
        stage_dependent_log_std: If True (requires state_dependent_log_std), use one
            log_std head per curriculum stage and select by ``task_stage`` in :meth:`act`.
        num_log_std_stages: Number of stage-specific log_std heads when
            stage_dependent_log_std is True (default: 1).
    """

    def __init__(
        self,
        z_dim,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        initial_log_std: float = 0,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        hiddens: list = [256, 128, 64],
        activations: list = ["elu", "elu", "elu", "tanh"],
        reduction: str = "sum",
        state_dependent_log_std: bool = False,
        stage_dependent_log_std: bool = False,
        num_log_std_stages: int = 1,
    ) -> None:
        super().__init__()

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.observation_space = observation_space
        self.action_space = action_space
        self._state_dependent_log_std = state_dependent_log_std
        if stage_dependent_log_std and not state_dependent_log_std:
            raise ValueError("stage_dependent_log_std requires state_dependent_log_std=True")
        self._stage_dependent_log_std = stage_dependent_log_std
        self._num_log_std_stages = num_log_std_stages

        num_actions = action_space.shape[0]

        # Build policy network
        hiddens = hiddens.copy()
        self.policy_net = MLP(z_dim, hiddens, activations[:-1]).to(device)

        # Mean head: Gain 0.01 for exploration
        self.mean_head = nn.Sequential(
            nn.Linear(hiddens[-1], num_actions),
            _ACTIVATIONS[activations[-1]] # Only add this if your actions are strictly [-1, 1]
        ).to(device)
        
        # Initialize log_std: parameter, single head, or one head per curriculum stage
        if state_dependent_log_std and stage_dependent_log_std:
            self.log_std_heads = nn.ModuleList(
                [nn.Linear(hiddens[-1], num_actions).to(device) for _ in range(num_log_std_stages)]
            )
        elif state_dependent_log_std:
            self.log_std_head = nn.Linear(hiddens[-1], num_actions).to(device)
        else:
            # Use learnable parameter
            self.log_std_parameter = nn.Parameter(
                initial_log_std * torch.ones(num_actions, device=device),
                requires_grad=True
            )

        # orthogonal initialization with gain 0.01 for the last layer
        self.policy_net.apply(init_ppo_weights)
        for layer in self.mean_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
        if state_dependent_log_std and stage_dependent_log_std:
            for head in self.log_std_heads:
                nn.init.orthogonal_(head.weight, gain=0.01)
                nn.init.constant_(head.bias, 0.0)
        elif state_dependent_log_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.constant_(self.log_std_head.bias, 0.0)

        # Action clipping setup
        self._clip_actions = clip_actions and (
            issubclass(type(self.action_space), gym.Space) or issubclass(type(self.action_space), gymnasium.Space)
        )
        if self._clip_actions:
            self._clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._clip_log_std = clip_log_std
        self._log_std_min = min_log_std
        self._log_std_max = max_log_std

        # Runtime state
        self._log_std = None
        self._num_samples = None
        self._distribution = None

        # Log probability reduction
        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError(f"reduction must be one of 'mean', 'sum', 'prod' or 'none', got '{reduction}'")
        self._reduction = {
            "mean": torch.mean,
            "sum": torch.sum,
            "prod": torch.prod,
            "none": None
        }[reduction]

    def _log_std_from_features(
        self, x: torch.Tensor, task_stage: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log_std from hidden features, optionally per curriculum stage."""
        if self._stage_dependent_log_std:
            if task_stage is None:
                raise ValueError("task_stage is required when stage_dependent_log_std=True")
            task_stage = task_stage.view(-1).long().to(device=x.device)
            if task_stage.shape[0] != x.shape[0]:
                raise ValueError(
                    f"task_stage batch size {task_stage.shape[0]} != features batch size {x.shape[0]}"
                )
            task_stage = task_stage.clamp(0, self._num_log_std_stages - 1)
            log_stds = torch.stack([head(x) for head in self.log_std_heads], dim=1)
            return log_stds[torch.arange(x.shape[0], device=x.device), task_stage]
        if self._state_dependent_log_std:
            return self.log_std_head(x)
        batch_size = x.shape[0]
        return self.log_std_parameter.unsqueeze(0).expand(batch_size, -1)

    def act(
        self,
        z,
        taken_actions=None,
        deterministic=False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Mapping[str, torch.Tensor]]:
        """Sample actions from the policy distribution.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            taken_actions: Pre-computed actions to evaluate log probability for (optional).
            deterministic: If True, return mean actions without sampling (default: False).
            
        Returns:
            Tuple of:
                - actions: Sampled actions of shape (batch_size, num_actions).
                - log_prob: Log probability of actions, or None if deterministic.
                - outputs: Dictionary containing "mean_actions".
                
        Example:
            >>> policy = GaussianPolicy(z_dim=64, action_space=action_space)
            >>> z = torch.randn(32, 64)
            >>> actions, log_prob, outputs = policy.act(z)
            >>> print(actions.shape, log_prob.shape)
            torch.Size([32, 8]) torch.Size([32, 1])
        """
        x = self.policy_net(z)
        mean_actions = self.mean_head(x)
        
        outputs = {}

        if deterministic:
            return mean_actions, None, outputs

        # Get and optionally clip log standard deviation
        log_std = self._log_std_from_features(x, task_stage=task_stage)
        
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        # Create normal distribution and sample using reparameterization trick
        self._distribution = Normal(mean_actions, log_std.exp())
        actions = self._distribution.rsample()

        # Compute log probability
        if taken_actions is None:
            taken_actions = actions
        log_prob = self._distribution.log_prob(taken_actions)
        
        # Reduce across action dimensions
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        # Clip actions to action space if enabled
        if self._clip_actions:
            actions = torch.clamp(actions, self._clip_actions_min, self._clip_actions_max)

        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    def get_per_stage_entropy_losses(
        self,
        entropy: torch.Tensor,
        task_stage: torch.Tensor,
        entropy_loss_scale: float,
    ) -> dict[int, float]:
        """Per-stage mean entropy loss using the active head's entropy from :meth:`act`."""
        if not self._stage_dependent_log_std or entropy_loss_scale <= 0:
            return {}
        entropy = entropy.reshape(-1)
        task_stage = task_stage.reshape(-1).long().to(device=entropy.device)
        if entropy.shape[0] != task_stage.shape[0]:
            raise ValueError(
                f"entropy has {entropy.shape[0]} samples but task_stage has {task_stage.shape[0]}"
            )
        losses = {}
        for stage_idx in range(self._num_log_std_stages):
            mask = task_stage == stage_idx
            if mask.any():
                losses[stage_idx] = (-entropy_loss_scale * entropy[mask].mean()).item()
        return losses

    def get_per_head_policy_stddev(self, z: torch.Tensor) -> dict[int, float]:
        """Mean policy stddev per log_std head (batch and action mean, logging only)."""
        if not self._stage_dependent_log_std:
            return {}
        with torch.no_grad():
            x = self.policy_net(z)
            stddev_by_head = {}
            for stage_idx, head in enumerate(self.log_std_heads):
                log_std = head(x)
                if self._clip_log_std:
                    log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
                stddev_by_head[stage_idx] = log_std.exp().mean().item()
            return stddev_by_head

    def get_per_stage_policy_stddev(self, task_stage: torch.Tensor) -> dict[int, float]:
        """Mean policy stddev per stage from the last :meth:`act` distribution (masked by task_stage)."""
        if self._distribution is None or not self._stage_dependent_log_std:
            return {}
        with torch.no_grad():
            stddev = self._distribution.stddev.mean(dim=-1).reshape(-1)
            task_stage = task_stage.reshape(-1).long().to(device=stddev.device)
            if stddev.shape[0] != task_stage.shape[0]:
                raise ValueError(
                    f"stddev has {stddev.shape[0]} samples but task_stage has {task_stage.shape[0]}"
                )
            stddev_by_stage = {}
            for stage_idx in range(self._num_log_std_stages):
                mask = task_stage == stage_idx
                if mask.any():
                    stddev_by_stage[stage_idx] = stddev[mask].mean().item()
            return stddev_by_stage

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Entropy of :attr:`_distribution` from the last :meth:`act` call.

        Uses the same action-dimension reduction as log-probability (e.g. sum).

        Args:
            role: Unused, kept for API compatibility.

        Returns:
            Per-sample entropy, shape ``(batch_size,)`` or ``(batch_size, 1)``.
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        #         return self._distribution.entropy().sum(dim=-1).unsqueeze(-1).to(self.device)
        # old
        # return self._distribution.entropy().to(self.device)
        # new
        return self._distribution.entropy().sum(dim=-1).unsqueeze(-1).to(self.device)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current action distribution.
        
        Args:
            role: Unused, kept for API compatibility.
            
        Returns:
            Normal distribution object.
        """
        return self._distribution


class DeterministicValue(torch.nn.Module):
    """Deterministic value function network.
    
    Maps latent representations to scalar value estimates using an MLP.
    
    Args:
        z_dim: Dimension of input latent representation.
        observation_space: Observation space (unused, kept for API compatibility).
        action_space: Action space (unused, kept for API compatibility).
        device: Device to run computations on.
        hiddens: List of hidden layer sizes (default: [256, 128, 64]).
        activations: List of activation function names (default: ["elu", "elu", "elu", "identity"]).
    """

    def __init__(
        self,
        z_dim,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        hiddens: list = [256, 128, 64],
        activations: list = ["elu", "elu", "elu", "identity"],
        scale_values: bool = True,
    ):
        super().__init__()

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )

        hiddens = hiddens.copy()
        hiddens.append(1)  # Output is scalar value
        self.value_net = MLP(z_dim, hiddens, activations).to(device)

        # Initialize state preprocessor if specified
        if scale_values:
            self.value_preprocessor = RunningStandardScaler(size=1, device=device)
        else:
            self.value_preprocessor = self.empty_preprocessor

        # 1. Initialize hidden layers with standard gain (sqrt(2) for ReLU/Tanh)
        self.value_net.apply(init_ppo_weights)

        # 2. Overwrite the final layer to use a gain of 1.0 (for the Critic)
        final_layer = list(self.value_net.modules())[-1] 
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=1.0)
            nn.init.constant_(final_layer.bias, 0.0)

    def compute_value(self, z, inverse=False) -> torch.Tensor:
        """Compute value estimate from latent representation.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            
        Returns:
            Value estimates of shape (batch_size, 1).
        """
        if inverse:
            return self.value_preprocessor(self.value_net(z), inverse=True)
        else:
            return self.value_net(z)

    def empty_preprocessor(self, x, train=False, inverse=False):
        return x


class MultiCritic(torch.nn.Module):
    """Wrapper for multiple value function networks (critics).
    
    Manages multiple critics and computes value estimates from each.
    Used for multi-critic PPO where advantages are computed separately
    for each critic and then combined.
    
    Args:
        critics: List of DeterministicValue networks.
    """
    
    def __init__(self, critics: List[DeterministicValue]):
        super().__init__()

       
        self.critics = torch.nn.ModuleList(critics)
        self.num_critics = len(critics)
        self.device = critics[0].device
        
    def compute_value(self, z, inverse=False) -> torch.Tensor:
        """Compute value estimates from all critics.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            
        Returns:
            Value estimates of shape (batch_size, num_critics).
            Each column corresponds to one critic's value estimate.
        """
        values = []
        for critic in self.critics:
            value = critic.compute_value(z, inverse=inverse)  # Shape: (batch_size, 1)
            values.append(value)
        # Stack along last dimension: (batch_size, num_critics)
        return torch.cat(values, dim=-1)

    def value_preprocessor(self, values, train=False, inverse=False):
        proccessed_values = []
        for i, critic in enumerate(self.critics):
            proccessed_values.append(critic.value_preprocessor(values[:, :, i].unsqueeze(-1), train=train, inverse=inverse))
        return torch.cat(proccessed_values, dim=-1)

    def parameters(self):
        return itertools.chain(*[critic.parameters() for critic in self.critics])
