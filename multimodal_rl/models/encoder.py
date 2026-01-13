import torch
import torch.nn as nn

from multimodal_rl.models.cnn import ImageEncoder
from multimodal_rl.models.mlp import MLP
from multimodal_rl.models.running_standard_scaler import RunningStandardScalerDict

methods = ["early", "intermediate"]

class Encoder(nn.Module):
    """Encoder handling early and intermediate fusion of raw and latent inputs."""

    def __init__(self, observation_space, action_space, env_cfg, config_dict, device):
        super().__init__()

        self.method = config_dict["encoder"]["method"]
        assert self.method in methods
        self.device = device

        self.observation_space = observation_space
        self.action_space = action_space

        self.num_inputs = 0
        self.hiddens = config_dict["encoder"]["hiddens"]
        self.activations = config_dict["encoder"]["activations"]
        
        # Latent dimension for state projections in intermediate fusion
        self.latent_state_dim = config_dict["encoder"].get("latent_state_dim", 64)

        # Standard scaler
        if config_dict["encoder"]["state_preprocessor"] is not None:
            self.state_preprocessor = RunningStandardScalerDict(size=observation_space, device=device)
        else:
            self.state_preprocessor = None

        # Dict to hold projection MLPs for intermediate fusion
        if self.method == "intermediate":
            print("Creating state encoders for intermediate fusion")
            self.state_encoders = nn.ModuleDict()
        else:
            self.state_encoders = None

        # Dict to hold CNNs for visual inputs (rgb and depth)
        self.cnns = nn.ModuleDict()

        # Configure relevant preprocessing
        # We sort keys to ensure deterministic ordering of the input vector
        for k in sorted(observation_space.keys()):
            v = observation_space[k]

            if k == "rgb" or k == "depth":
                latent_pixel_dim = config_dict["observations"]["pixel_cfg"]["latent_pixel_dim"]
                # Create separate CNN for each visual input type
                self.cnns[k] = ImageEncoder(v.shape, latent_pixel_dim).to(device)
                self.num_inputs += latent_pixel_dim

            elif k in ["prop", "gt", "tactile"]:
                input_dim = v.shape[0]
                
                if self.method == "early":
                    self.num_inputs += input_dim
                
                elif self.method == "intermediate":
                    # Create a small MLP to project raw state to a latent vector
                    # Using a single hidden layer or direct projection
                    projection_mlp = MLP(
                        input_dim, 
                        [self.latent_state_dim], 
                        self.activations, 
                        layernorm=config_dict["encoder"]["layernorm"]
                    ).to(device)
                    print(f"Created projection MLP for '{k}' with input dim {input_dim} to latent dim {self.latent_state_dim}")
                    self.state_encoders[k] = projection_mlp
                    self.num_inputs += self.latent_state_dim

        self.num_outputs = self.hiddens[-1]
        self.net = MLP(
            self.num_inputs, 
            self.hiddens, 
            self.activations, 
            layernorm=config_dict["encoder"]["layernorm"]
        ).to(device)

    def forward(self, obs_dict, detach=False, train=False):
        # Handle potential nesting
        if "policy" in obs_dict.keys():
            obs_dict = obs_dict["policy"]

        if detach:
            obs_dict = {key: value.detach() for key, value in obs_dict.items()}

        # Scale inputs
        if self.state_preprocessor is not None:
            obs_dict = self.state_preprocessor(obs_dict, train)

        # 1. Get Latent Visual Inputs (always processed via CNN)
        latent_visuals = self.get_latent_inputs(obs_dict)

        # 2. Process State Inputs based on method
        if self.method == "early":
            # Just concatenate raw tensors
            raw_states = self.get_raw_inputs(obs_dict)
            concat_obs = torch.cat((raw_states, latent_visuals), dim=-1)
        
        elif self.method == "intermediate":
            # Pass each state through its specific projection MLP
            processed_states = []
            for k in sorted(obs_dict.keys()):
                if k in self.state_encoders:
                    z_state = self.state_encoders[k](obs_dict[k][:])
                    processed_states.append(z_state)
            
            # Concatenate projected states with visual latents
            concat_obs = torch.cat(processed_states + [latent_visuals], dim=-1)

        # Final MLP head
        z = self.net(concat_obs)

        if z.dim() == 1:
            z = z.unsqueeze(0) 

        return z

    def get_raw_inputs(self, obs_dict):
        """Retrieve and concat prop, gt, or tactile raw values."""
        raw_inputs = []
        for k in sorted(obs_dict.keys()):
            if k in ["prop", "gt", "tactile"]:
                raw_inputs.append(obs_dict[k][:])
        return torch.cat(raw_inputs, dim=-1) if raw_inputs else torch.tensor([]).to(self.device)

    def get_latent_inputs(self, obs_dict):
        """Pass visual inputs through CNN."""
        latent_inputs = []
        for k in sorted(obs_dict.keys()):
            if k == "rgb" or k == "depth":
                z = self.cnns[k](obs_dict[k][:])
                latent_inputs.append(z)
        return torch.cat(latent_inputs, dim=-1) if latent_inputs else torch.tensor([]).to(self.device)