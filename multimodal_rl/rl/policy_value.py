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
        activations: List of activation function names. With a single global mean
            head, this includes the final action activation. With
            stage_dependent_mean=True, this configures only the shared policy
            hidden layers; the final action activation belongs to
            stage_mean_activations.
        reduction: How to reduce log probability across action dimensions:
            "sum", "mean", "prod", or "none" (default: "sum").
        stage_dependent_mean: If True, use one mean head per curriculum stage and
            select by ``task_stage`` in :meth:`act`.
        num_mean_stages: Number of stage-specific mean heads when
            stage_dependent_mean is True (default: 1).
        stage_mean_hiddens: Hidden layer sizes for each stage-specific mean head.
        stage_mean_activations: Activations for each stage-specific mean head,
            including the final action activation.
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
        stage_dependent_mean: bool = False,
        num_mean_stages: int = 1,
        stage_mean_hiddens: Optional[List[int]] = None,
        stage_mean_activations: Optional[List[str]] = None,
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
        self._stage_dependent_mean = stage_dependent_mean
        self._num_mean_stages = num_mean_stages
        self._state_dependent_log_std = state_dependent_log_std
        if stage_dependent_log_std and not state_dependent_log_std:
            raise ValueError("stage_dependent_log_std requires state_dependent_log_std=True")
        self._stage_dependent_log_std = stage_dependent_log_std
        self._num_log_std_stages = num_log_std_stages

        num_actions = action_space.shape[0]
        self.num_actions = num_actions

        # Build shared policy network. The old single-head path keeps the final
        # action activation in ``activations``; stage-specific mean heads own
        # their final activation via ``stage_mean_activations``.
        hiddens = hiddens.copy()
        if stage_dependent_mean:
            if stage_mean_activations is None and len(activations) == len(hiddens) + 1:
                shared_activations = activations[:-1]
                stage_mean_activations = [activations[-1]]
            else:
                shared_activations = activations
        else:
            shared_activations = activations[:-1]
        self.policy_net = MLP(z_dim, hiddens, shared_activations).to(device)
        policy_output_dim = hiddens[-1] if hiddens else z_dim

        # Mean head(s): shared policy features followed by either one global head
        # or one stage-specific head selected by task_stage.
        if stage_dependent_mean:
            if num_mean_stages < 1:
                raise ValueError("num_mean_stages must be >= 1")
            stage_mean_hiddens = [] if stage_mean_hiddens is None else stage_mean_hiddens.copy()
            stage_mean_activations = (
                ["tanh"]
                if stage_mean_activations is None
                else stage_mean_activations.copy()
            )
            self.mean_heads = nn.ModuleList(
                [
                    self._make_mean_head(
                        policy_output_dim,
                        num_actions,
                        stage_mean_hiddens,
                        stage_mean_activations,
                    ).to(device)
                    for _ in range(num_mean_stages)
                ]
            )
        else:
            self.mean_head = self._make_mean_head(
                policy_output_dim,
                num_actions,
                [],
                [activations[-1]],
            ).to(device)
        
        # Initialize log_std: parameter, single head, or one head per curriculum stage
        if state_dependent_log_std and stage_dependent_log_std:
            self.log_std_heads = nn.ModuleList(
                [nn.Linear(policy_output_dim, num_actions).to(device) for _ in range(num_log_std_stages)]
            )
        elif state_dependent_log_std:
            self.log_std_head = nn.Linear(policy_output_dim, num_actions).to(device)
        else:
            # Use learnable parameter
            self.log_std_parameter = nn.Parameter(
                initial_log_std * torch.ones(num_actions, device=device),
                requires_grad=True
            )

        # orthogonal initialization with gain 0.01 for the last layer
        self.policy_net.apply(init_ppo_weights)
        if stage_dependent_mean:
            for head in self.mean_heads:
                self._init_mean_head(head)
        else:
            self._init_mean_head(self.mean_head)
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

    def _make_mean_head(
        self,
        input_dim: int,
        num_actions: int,
        hiddens: List[int],
        activations: List[str],
    ) -> nn.Sequential:
        """Build a mean head with optional private hidden layers."""
        if len(activations) != len(hiddens) + 1:
            raise ValueError(
                "stage_mean_activations must have one activation per hidden layer "
                "plus one final action activation"
            )
        layers = []
        if hiddens:
            # Use _modules.values() rather than children(): children() de-duplicates
            # repeated activation module instances, which can silently drop ELUs.
            layers.extend(MLP(input_dim, hiddens, activations[:-1])._modules.values())
            input_dim = hiddens[-1]
        layers.append(nn.Linear(input_dim, num_actions))
        layers.append(_ACTIVATIONS[activations[-1]])
        return nn.Sequential(*layers)

    def _init_mean_head(self, head: nn.Sequential) -> None:
        """Initialize private head hidden layers normally and final mean layer gently."""
        linear_layers = [module for module in head.modules() if isinstance(module, nn.Linear)]
        for layer in linear_layers[:-1]:
            init_ppo_weights(layer)
        if linear_layers:
            nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)
            nn.init.constant_(linear_layers[-1].bias, 0.0)

    def _mean_from_features(
        self, x: torch.Tensor, task_stage: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mean actions from shared policy features, optionally per stage."""
        if self._stage_dependent_mean:
            if task_stage is None:
                raise ValueError("task_stage is required when stage_dependent_mean=True")
            task_stage = task_stage.view(-1).long().to(device=x.device)
            if task_stage.shape[0] != x.shape[0]:
                raise ValueError(
                    f"task_stage batch size {task_stage.shape[0]} != features batch size {x.shape[0]}"
                )
            task_stage = task_stage.clamp(0, self._num_mean_stages - 1)
            mean_actions = x.new_empty((x.shape[0], self.num_actions))
            for stage_idx, head in enumerate(self.mean_heads):
                mask = task_stage == stage_idx
                if mask.any():
                    mean_actions[mask] = head(x[mask])
            return mean_actions
        return self.mean_head(x)

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
            log_std = x.new_empty((x.shape[0], self.num_actions))
            for stage_idx, head in enumerate(self.log_std_heads):
                mask = task_stage == stage_idx
                if mask.any():
                    log_std[mask] = head(x[mask])
            return log_std
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
        mean_actions = self._mean_from_features(x, task_stage=task_stage)
        
        outputs = {"mean_actions": mean_actions}

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
        entropy = self._distribution.entropy().sum(dim=-1).unsqueeze(-1).to(self.device)

        # print(f"summed entropy across actions: {entropy.shape}, {entropy.min()}, {entropy.max()}")
        return entropy

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current action distribution.
        
        Args:
            role: Unused, kept for API compatibility.
            
        Returns:
            Normal distribution object.
        """
        return self._distribution


class StageValuePreprocessor(nn.Module):
    """One running value scaler per curriculum stage."""

    def __init__(self, num_stages: int, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        self.scalers = nn.ModuleList(
            [RunningStandardScaler(size=1, device=device) for _ in range(num_stages)]
        )
        self.running_mean_mean = 0.0
        self.running_variance_mean = 1.0

    def _refresh_stats(self) -> None:
        self.running_mean_mean = float(
            np.mean([scaler.running_mean_mean for scaler in self.scalers])
        )
        self.running_variance_mean = float(
            np.mean([scaler.running_variance_mean for scaler in self.scalers])
        )

    def forward(
        self,
        values: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if task_stage is None:
            raise ValueError("task_stage is required for stage-dependent value preprocessing")
        if values.shape[-1] != 1:
            raise ValueError(f"Expected scalar values with trailing dim 1, got {values.shape}")

        original_shape = values.shape
        flat_values = values.reshape(-1, values.shape[-1])
        flat_stage = task_stage.reshape(-1).long().to(device=values.device)
        if flat_stage.shape[0] != flat_values.shape[0]:
            raise ValueError(
                f"task_stage has {flat_stage.shape[0]} samples but values has {flat_values.shape[0]}"
            )
        flat_stage = flat_stage.clamp(0, len(self.scalers) - 1)

        processed = flat_values.new_empty(flat_values.shape)
        for stage_idx, scaler in enumerate(self.scalers):
            mask = flat_stage == stage_idx
            if mask.any():
                stage_values = flat_values[mask]
                # RunningStandardScaler uses unbiased variance; updating on one
                # sample would produce NaNs, so keep existing stats in that case.
                update_stats = train and stage_values.shape[0] > 1
                processed[mask] = scaler(stage_values, train=update_stats, inverse=inverse)

        self._refresh_stats()
        return processed.reshape(original_shape)


class DeterministicValue(torch.nn.Module):
    """Deterministic value function network.
    
    Maps latent representations to scalar value estimates using an MLP.
    
    Args:
        z_dim: Dimension of input latent representation.
        observation_space: Observation space (unused, kept for API compatibility).
        action_space: Action space (unused, kept for API compatibility).
        device: Device to run computations on.
        hiddens: List of hidden layer sizes (default: [256, 128, 64]).
        activations: List of activation function names. For the default single
            value network this includes the final value activation. With
            stage_dependent_value=True and shared_stage_value=True, this configures
            only the shared value trunk.
        stage_dependent_value: If True, select one value head/network per
            curriculum stage using ``task_stage`` in :meth:`compute_value`.
        shared_stage_value: If True, use a shared value trunk plus one head per
            stage. If False, use one complete value network per stage.
        num_value_stages: Number of stage-specific value heads/networks.
        stage_value_hiddens: Hidden layer sizes for each stage-specific value head.
        stage_value_activations: Activations for each stage-specific value head,
            including the final value activation.
    """

    def __init__(
        self,
        z_dim,
        observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        hiddens: list = [256, 128, 64],
        activations: list = ["elu", "elu", "elu", "identity"],
        stage_dependent_value: bool = False,
        shared_stage_value: bool = True,
        num_value_stages: int = 1,
        stage_value_hiddens: Optional[List[int]] = None,
        stage_value_activations: Optional[List[str]] = None,
        value_preprocessor_mode: str = "per_stage",
        scale_values: bool = True,
    ):
        super().__init__()

        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self._stage_dependent_value = stage_dependent_value
        self._shared_stage_value = shared_stage_value
        self._num_value_stages = num_value_stages
        self._value_preprocessor_mode = value_preprocessor_mode

        hiddens = hiddens.copy()
        if stage_dependent_value:
            if num_value_stages < 1:
                raise ValueError("num_value_stages must be >= 1")
            if shared_stage_value:
                if stage_value_activations is None and len(activations) == len(hiddens) + 1:
                    shared_activations = activations[:-1]
                    stage_value_activations = [activations[-1]]
                else:
                    shared_activations = activations
                stage_value_hiddens = [] if stage_value_hiddens is None else stage_value_hiddens.copy()
                stage_value_activations = (
                    ["identity"]
                    if stage_value_activations is None
                    else stage_value_activations.copy()
                )
                self.value_net = MLP(z_dim, hiddens, shared_activations).to(device)
                value_output_dim = hiddens[-1] if hiddens else z_dim
                self.value_heads = nn.ModuleList(
                    [
                        self._make_value_head(
                            value_output_dim,
                            stage_value_hiddens,
                            stage_value_activations,
                        ).to(device)
                        for _ in range(num_value_stages)
                    ]
                )
            else:
                full_hiddens = hiddens.copy()
                full_hiddens.append(1)
                self.value_nets = nn.ModuleList(
                    [MLP(z_dim, full_hiddens, activations).to(device) for _ in range(num_value_stages)]
                )
        else:
            hiddens.append(1)  # Output is scalar value
            self.value_net = MLP(z_dim, hiddens, activations).to(device)

        # Initialize value scaler(s) if specified.
        # When stage-dependent value heads are enabled you can choose whether the
        # running value normalization statistics are shared across stages or per-stage.
        if scale_values:
            if stage_dependent_value and value_preprocessor_mode not in ("per_stage", "shared"):
                raise ValueError(
                    f"value_preprocessor_mode must be 'per_stage' or 'shared', got '{value_preprocessor_mode}'"
                )
            if stage_dependent_value and value_preprocessor_mode == "per_stage":
                self.value_preprocessor = StageValuePreprocessor(num_stages=num_value_stages, device=device)
            else:
                self.value_preprocessor = RunningStandardScaler(size=1, device=device)
        else:
            self.value_preprocessor = self.empty_preprocessor

        if stage_dependent_value and shared_stage_value:
            self.value_net.apply(init_ppo_weights)
            for head in self.value_heads:
                self._init_value_net(head)
        elif stage_dependent_value:
            for value_net in self.value_nets:
                self._init_value_net(value_net)
        else:
            self._init_value_net(self.value_net)

    def _make_value_head(
        self,
        input_dim: int,
        hiddens: List[int],
        activations: List[str],
    ) -> nn.Sequential:
        """Build a stage-specific value head with optional private hidden layers."""
        if len(activations) != len(hiddens) + 1:
            raise ValueError(
                "stage_value_activations must have one activation per hidden layer "
                "plus one final value activation"
            )
        layers = []
        if hiddens:
            layers.extend(MLP(input_dim, hiddens, activations[:-1])._modules.values())
            input_dim = hiddens[-1]
        layers.append(nn.Linear(input_dim, 1))
        layers.append(_ACTIVATIONS[activations[-1]])
        return nn.Sequential(*layers)

    def _init_value_net(self, value_net: nn.Module) -> None:
        """Initialize value hidden layers normally and the final value layer with gain 1."""
        value_net.apply(init_ppo_weights)
        linear_layers = [module for module in value_net.modules() if isinstance(module, nn.Linear)]
        if linear_layers:
            nn.init.orthogonal_(linear_layers[-1].weight, gain=1.0)
            nn.init.constant_(linear_layers[-1].bias, 0.0)

    def _value_from_features(
        self, z: torch.Tensor, task_stage: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute raw value predictions, optionally selecting a stage-specific path."""
        if self._stage_dependent_value:
            if task_stage is None:
                raise ValueError("task_stage is required when stage_dependent_value=True")
            task_stage = task_stage.view(-1).long().to(device=z.device)
            if task_stage.shape[0] != z.shape[0]:
                raise ValueError(
                    f"task_stage batch size {task_stage.shape[0]} != features batch size {z.shape[0]}"
                )
            task_stage = task_stage.clamp(0, self._num_value_stages - 1)
            values = z.new_empty((z.shape[0], 1))
            if self._shared_stage_value:
                x = self.value_net(z)
                for stage_idx, head in enumerate(self.value_heads):
                    mask = task_stage == stage_idx
                    if mask.any():
                        values[mask] = head(x[mask])
            else:
                for stage_idx, value_net in enumerate(self.value_nets):
                    mask = task_stage == stage_idx
                    if mask.any():
                        values[mask] = value_net(z[mask])
            return values
        return self.value_net(z)

    def compute_value(
        self,
        z,
        inverse=False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value estimate from latent representation.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            
        Returns:
            Value estimates of shape (batch_size, 1).
        """
        values = self._value_from_features(z, task_stage=task_stage)
        if inverse:
            return self.preprocess_values(values, inverse=True, task_stage=task_stage)
        else:
            return values

    def preprocess_values(
        self,
        values: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._stage_dependent_value and isinstance(self.value_preprocessor, StageValuePreprocessor):
            return self.value_preprocessor(
                values, train=train, inverse=inverse, task_stage=task_stage
            )
        return self.value_preprocessor(values, train=train, inverse=inverse)

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
        self._stage_dependent_value = any(
            getattr(critic, "_stage_dependent_value", False) for critic in critics
        )
        self._num_value_stages = max(
            getattr(critic, "_num_value_stages", 1) for critic in critics
        )
        
    def compute_value(
        self,
        z,
        inverse=False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value estimates from all critics.
        
        Args:
            z: Latent representation tensor of shape (batch_size, z_dim).
            
        Returns:
            Value estimates of shape (batch_size, num_critics).
            Each column corresponds to one critic's value estimate.
        """
        values = []
        for critic in self.critics:
            value = critic.compute_value(
                z, inverse=inverse, task_stage=task_stage
            )  # Shape: (batch_size, 1)
            values.append(value)
        # Stack along last dimension: (batch_size, num_critics)
        return torch.cat(values, dim=-1)

    def preprocess_values(
        self,
        values: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        proccessed_values = []
        for i, critic in enumerate(self.critics):
            proccessed_values.append(
                critic.preprocess_values(
                    values[..., i].unsqueeze(-1),
                    train=train,
                    inverse=inverse,
                    task_stage=task_stage,
                )
            )
        return torch.cat(proccessed_values, dim=-1)

    def value_preprocessor(
        self,
        values: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        task_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.preprocess_values(
            values, train=train, inverse=inverse, task_stage=task_stage
        )

    def parameters(self):
        return itertools.chain(*[critic.parameters() for critic in self.critics])
