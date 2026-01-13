"""Policy and value network implementations for PPO.

Provides Gaussian policy (stochastic) and deterministic value function networks.
"""

import gym
import gymnasium
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any, Mapping, Optional, Tuple, Union

from multimodal_rl.models.mlp import MLP

_ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


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
    ) -> None:
        super().__init__()

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.observation_space = observation_space
        self.action_space = action_space

        num_actions = action_space.shape[0]
        self.log_std_parameter = nn.Parameter(
            initial_log_std * torch.ones(num_actions, device=device),
            requires_grad=True
        )

        # Build policy network
        if not hiddens:
            self.policy_net = nn.Sequential(nn.Linear(z_dim, num_actions), _ACTIVATIONS[activations[-1]])
        else:
            hiddens = hiddens.copy()
            hiddens.append(num_actions)
            self.policy_net = MLP(z_dim, hiddens, activations).to(device)

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

    def act(
        self, z, taken_actions=None, deterministic=False
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
        mean_actions = self.policy_net(z)
        outputs = {}

        if deterministic:
            return mean_actions, None, outputs

        # Get and optionally clip log standard deviation
        log_std = self.log_std_parameter
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

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute entropy of the current policy distribution.
        
        Args:
            role: Unused, kept for API compatibility.
            
        Returns:
            Entropy tensor of shape (batch_size, num_actions).
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Get log standard deviation for current batch.
        
        Args:
            role: Unused, kept for API compatibility.
            
        Returns:
            Log standard deviation tensor of shape (batch_size, num_actions).
        """
        return self._log_std.repeat(self._num_samples, 1)

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
    ):
        super().__init__()

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )

        hiddens = hiddens.copy()
        hiddens.append(1)  # Output is scalar value
        self.value_net = MLP(z_dim, hiddens, activations).to(device)

    def compute_value(self, z) -> torch.Tensor:
        """Compute value estimate from latent representation.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            
        Returns:
            Value estimates of shape (batch_size, 1).
        """
        return self.value_net(z)
