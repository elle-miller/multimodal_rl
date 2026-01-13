import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_rl.ssl.task import AuxiliaryTask
from multimodal_rl.wrappers.frame_stack import LazyFrames


class CustomDecoder(nn.Module):
    """
    Decoder to map the latent representation (256 dim) back to the input space.
    Uses a fixed, high-capacity architecture (256 -> 512 -> 1024) to maximize
    expressiveness, regardless of computational cost.
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super(CustomDecoder, self).__init__()
        
        # Use the high-capacity, fixed architecture (256 -> 512 -> 1024)
        # Note: output_dim is used to correctly size the final layer, 
        # but the preceding layers are fixed for max capacity.
        self.net = nn.Sequential(
            # Input: Latent dim (256)
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(alpha=1.0),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ELU(alpha=1.0),
            
            # Output: Final dimension determined by the task (e.g., 68 for tactile, 768 for full)
            nn.Linear(1024, output_dim),
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.
        Returns logits (raw outputs) suitable for BCEWithLogitsLoss.
        """
        return self.net(x)
        
         

class Reconstruction(AuxiliaryTask):
    """
    Auxiliary task for state reconstruction (Tactile or Full).
    Aims to force the encoder to learn an information-rich latent representation.
    """

    def __init__(self, aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer):
        super().__init__(aux_task_cfg, rl_rollout, rl_memory, encoder, value, value_preprocessor, env, env_cfg, writer)

        self.num_prop_obs = env.observation_space["policy"]["prop"].shape[0]
        self.num_tactile_obs = env.observation_space["policy"]["tactile"].shape[0]
        self.is_binary_tactile = env.binary_tactile # Renamed for clarity

        print("****Reconstruction Network Configuration*****")
        print(f"Latent dim: {self.encoder.num_outputs}") # Should be 256
        print(f"Tactile size: {self.num_tactile_obs}")
        print(f"Prop size: {self.num_prop_obs}")
        
        # Determine the total output dimension required
        if self.tactile_only:
            output_dim = self.num_tactile_obs
        else:
            output_dim = self.num_prop_obs + self.num_tactile_obs
        
        # Use the fixed, high-capacity decoder
        self.decoder = CustomDecoder(
            latent_dim=self.encoder.num_outputs, # 256
            output_dim=output_dim
        ).to(self.device)
        print(self.decoder)

        # Initialize tracking tensors on device
        self.mean_tactile = torch.zeros((self.num_tactile_obs,)).to(self.device)
        self.pos_weight = torch.ones((self.num_tactile_obs,)).to(self.device)

        super()._post_init()

    # --- Override methods for AuxiliaryTask ---

    def set_optimisable_networks(self):
        # Ensure both the shared encoder and the decoder are trained
        return [self.encoder, self.decoder]

    def create_memory(self):
        # Use parent implementation or leave blank if memory is inherited
        pass

    def sample_minibatches(self, minibatches):
        # A clearer, simpler way to sample from the main RL memory
        sampled_batches = self.memory.sample_all(mini_batches=minibatches)
        return [sampled_batches] # Return as a list of batches
    
    def compute_loss(self, minibatch):
        """
        Compute the Reconstruction loss (Tactile BCE + Prop MSE).
        """
        states, _ = minibatch # Actions are not needed for reconstruction
        states, next_states = self.separate_memory_tensors(states)
        obs_dict = states["policy"]

        # --- Prepare Ground Truth ---
        # Note: If tactile is not frame-stacked, you can simplify the indexing. 
        # Assuming the observation is the current state (time t).
        
        # Proprioception: Assumed to be continuous values (MSE)
        prop_true = obs_dict["prop"][:]

        # Tactile: Assumed to be binary (BCE with Logits)
        tactile_true = obs_dict["tactile"][:]

        # --- Compute Adaptive Positive Weight for BCE ---
        # Calculate positive weight based on the current minibatch to handle class imbalance
        minibatch_tactile_mean = torch.mean(tactile_true, dim=0) # Mean of 1s across batch
        # Clamp to a minimum value (e.g., 1e-6) to avoid division by zero/NaNs if a tactile sensor is always 0
        minibatch_tactile_mean = torch.clamp(minibatch_tactile_mean, min=1e-6)
        
        # Weight = (1 - mean) / mean. This is the standard ratio for imbalance.
        # Ratio of zeros to ones.
        pos_weight = (1.0 - minibatch_tactile_mean) / minibatch_tactile_mean

        # Clamp to avoid extreme values, e.g., if mean is very close to 0. 
        # (Your max=10000.0 is a reasonable heuristic.)
        pos_weight = torch.clamp(pos_weight, max=10000.0)
        
        # --- Compute Prediction ---
        z = self.encoder(obs_dict)
        x_hat = self.decoder(z) # x_hat contains raw logits

        if self.tactile_only:
            tactile_pred_logits = x_hat
            prop_loss = torch.tensor(0.0, device=self.device)
        else:
            # Full Reconstruction: Split the prediction
            tactile_pred_logits = x_hat[:, self.num_prop_obs:]
            prop_pred = x_hat[:, :self.num_prop_obs]
            
            # Proprioception Loss (Continuous data -> MSE)
            prop_loss = F.mse_loss(prop_pred, prop_true)

        # Tactile Loss (Binary data -> BCE with Logits)
        # The loss applies the pos_weight vector element-wise to handle sensor imbalance.
        tactile_loss = F.binary_cross_entropy_with_logits(
            tactile_pred_logits, 
            tactile_true, 
            pos_weight=pos_weight # Per-element weight
        )

        loss = tactile_loss + prop_loss

        # --- Logging and Metrics ---
        info = {
            "Loss/Recon_total_loss": loss.item(),
            "Loss/Recon_tactile_loss": tactile_loss.item(),
            "Metrics/Avg_pos_weight": pos_weight.mean().item()
        }
        
        if self.is_binary_tactile:
            # Need to convert logits to probabilities, then to predictions (0/1) for evaluation
            tactile_pred_prob = torch.sigmoid(tactile_pred_logits)
            more_info = self.evaluate_binary_predictions(tactile_pred_prob, tactile_true)
            info.update(more_info)
            
        if not self.tactile_only:
            info["Loss/Recon_prop_loss"] = prop_loss.item()

        return loss, info
    


    def add_samples(self, states):
        """
        Add samples to dedicated aux memory
        Re-implement this if you don't need all of these tensors

        Samples must come in as Lazy Frames

        But be saved in their expanded form
        
        """
        if not isinstance(states["policy"]["prop"], LazyFrames):
            raise TypeError("should be LazyFrames")
        
        states = states["policy"]

        # don't need to worry about alive/dead for reconstruction
        for obs_k in self.env.observation_space["policy"].keys():
            if isinstance(states[obs_k], LazyFrames):
                states[obs_k] = states[obs_k][:]
