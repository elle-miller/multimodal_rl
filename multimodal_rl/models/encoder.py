"""Multimodal encoder for fusing diverse observation types.

Supports early fusion (concatenation) and intermediate fusion (projection) strategies
for combining visual (RGB, depth) and state-based (proprioception, tactile, ground-truth) observations.
"""

import torch
import torch.nn as nn

from multimodal_rl.models.cnn import ImageEncoder
from multimodal_rl.models.mlp import MLP
from multimodal_rl.models.running_standard_scaler import RunningStandardScalerDict

FUSION_METHODS = ["early", "intermediate"]


class Encoder(nn.Module):
    """Multimodal encoder supporting early and intermediate fusion strategies.
    
    Early fusion: Raw state observations are concatenated directly with visual latents.
    Intermediate fusion: State observations are projected to a latent space before fusion.
    
    Args:
        observation_space: Dictionary mapping observation names to their shapes.
        action_space: Action space (unused but kept for API compatibility).
        env_cfg: Environment configuration.
        config_dict: Configuration dictionary containing encoder settings:
            - method: Fusion method ("early" or "intermediate")
            - hiddens: List of hidden layer sizes for final MLP
            - activations: List of activation functions
            - layernorm: Whether to use layer normalization
            - latent_state_dim: Latent dimension for intermediate fusion (default: 64)
            - state_preprocessor: Whether to use state preprocessing
        device: Device to run computations on.
    """

    def __init__(self, observation_space, action_space, env_cfg, config_dict, device):
        super().__init__()
        
        self.method = config_dict["encoder"]["method"]
        if self.method not in FUSION_METHODS:
            raise ValueError(f"Unknown fusion method: {self.method}. Must be one of {FUSION_METHODS}")
        
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.hiddens = config_dict["encoder"]["hiddens"]
        self.activations = config_dict["encoder"]["activations"]
        self.latent_state_dim = config_dict["encoder"].get("latent_state_dim", 64)

        # Initialize state preprocessor if specified
        if config_dict["encoder"]["state_preprocessor"] is not None:
            self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)
        else:
            self.state_preprocessor = None

        # Initialize encoders based on fusion method
        self.state_encoders = nn.ModuleDict() if self.method == "intermediate" else None
        self.cnns = nn.ModuleDict()
        
        num_inputs = 0
        
        # Build encoders for each observation type
        # Sort keys for deterministic ordering
        for k in sorted(observation_space.keys()):
            v = observation_space[k]

            if k in ("rgb", "depth"):
                # Visual inputs: process through CNN
                latent_pixel_dim = config_dict["observations"]["pixel_cfg"]["latent_pixel_dim"]
                self.cnns[k] = ImageEncoder(v.shape, latent_pixel_dim).to(device)
                num_inputs += latent_pixel_dim

            elif k in ("prop", "gt", "tactile"):
                # State inputs: process based on fusion method
                input_dim = v.shape[0]
                
                if self.method == "early":
                    num_inputs += input_dim
                elif self.method == "intermediate":
                    # Project state to latent space
                    projection_mlp = MLP(
                        input_dim,
                        [self.latent_state_dim],
                        self.activations,
                        layernorm=config_dict["encoder"]["layernorm"]
                    ).to(device)
                    self.state_encoders[k] = projection_mlp
                    num_inputs += self.latent_state_dim

        self.num_inputs = num_inputs
        self.num_outputs = self.hiddens[-1]
        
        # Final fusion MLP
        self.net = MLP(
            num_inputs,
            self.hiddens,
            self.activations,
            layernorm=config_dict["encoder"]["layernorm"]
        ).to(device)

    def forward(self, obs_dict, detach=False, train=False):
        """Forward pass through the encoder.
        
        Args:
            obs_dict: Dictionary of observations, optionally nested under "policy" key.
            detach: If True, detach gradients from inputs.
            train: If True, update running statistics for preprocessing.
            
        Returns:
            Encoded representation tensor of shape (batch_size, num_outputs).
        """
        # Handle nested observation dictionaries
        if "policy" in obs_dict:
            obs_dict = obs_dict["policy"]

        if detach:
            obs_dict = {key: value.detach() for key, value in obs_dict.items()}

        # Preprocess observations if scaler is available
        if self.state_preprocessor is not None:
            obs_dict = self.state_preprocessor(obs_dict, train)

        # Process visual inputs through CNNs
        latent_visuals = self._get_latent_visuals(obs_dict)

        # Process state inputs based on fusion method
        if self.method == "early":
            raw_states = self._get_raw_states(obs_dict)
            concat_obs = torch.cat((raw_states, latent_visuals), dim=-1)
        elif self.method == "intermediate":
            processed_states = []
            for k in sorted(obs_dict.keys()):
                if k in self.state_encoders:
                    z_state = self.state_encoders[k](obs_dict[k][:])
                    processed_states.append(z_state)
            concat_obs = torch.cat(processed_states + [latent_visuals], dim=-1)

        # Final MLP projection
        z = self.net(concat_obs)

        # Ensure batch dimension exists
        if z.dim() == 1:
            z = z.unsqueeze(0)

        return z

    def _get_raw_states(self, obs_dict):
        """Extract and concatenate raw state observations.
        
        Args:
            obs_dict: Dictionary of observations.
            
        Returns:
            Concatenated tensor of raw state observations.
        """
        raw_inputs = []
        for k in sorted(obs_dict.keys()):
            if k in ("prop", "gt", "tactile"):
                raw_inputs.append(obs_dict[k][:])
        return torch.cat(raw_inputs, dim=-1) if raw_inputs else torch.tensor([], device=self.device)

    def _get_latent_visuals(self, obs_dict):
        """Process visual observations through CNNs.
        
        Args:
            obs_dict: Dictionary of observations.
            
        Returns:
            Concatenated tensor of visual latent representations.
        """
        latent_inputs = []
        for k in sorted(obs_dict.keys()):
            if k in ("rgb", "depth"):
                z = self.cnns[k](obs_dict[k][:])
                latent_inputs.append(z)
        return torch.cat(latent_inputs, dim=-1) if latent_inputs else torch.tensor([], device=self.device)