"""Convolutional neural network modules for image-based observations.

Based on architectures from:
- https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
- https://github.com/uoe-agents/MVD/blob/main/utils.py
"""

import numpy as np
import torch
import torch.nn as nn
from multimodal_rl.wrappers.frame_stack import LazyFrames


def weight_init(m):
    """Custom weight initialization for Conv2D and Linear layers.
    
    Uses orthogonal initialization for linear layers and delta-orthogonal
    initialization for convolutional layers (see arXiv:1806.05393).
    
    Args:
        m: Module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Delta-orthogonal initialization for convolutional layers
        assert m.weight.size(2) == m.weight.size(3), "Kernel must be square"
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class ImageEncoder(nn.Module):
    """Convolutional encoder for image-based observations.
    
    Architecture: 3-layer CNN (32->64->64 channels) with ReLU activations,
    followed by a linear projection to latent space.
    
    Supports both channels-first (C, H, W) and channels-last (H, W, C) input formats.
    Automatically normalizes uint8 inputs to [0, 1] range.
    
    Args:
        obs_shape: Observation shape tuple. Can be (C, H, W) or (H, W, C).
        latent_dim: Dimension of output latent representation (default: 50).
        num_layers: Number of convolutional layers (unused, kept for compatibility).
        num_filters: Number of filters in first layer (unused, kept for compatibility).
    """

    def __init__(self, obs_shape, latent_dim=50, num_layers=4, num_filters=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Determine channel location dynamically
        self.channels_first = np.argmin(obs_shape) == 0
        self.num_channels = obs_shape[0] if self.channels_first else obs_shape[-1]
        self.img_dim = obs_shape[1]

        # Convolutional layers: progressively downsample and increase channels
        # Layer 1: 8x8 kernel, stride 4 -> 32 channels, ~4x downsampling
        # Layer 2: 4x4 kernel, stride 2 -> 64 channels, ~2x downsampling  
        # Layer 3: 3x3 kernel, stride 1 -> 64 channels, same size
        self.convs = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # Compute flattened size dynamically based on input resolution
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_channels, self.img_dim, self.img_dim)
            conv_out = self.convs(dummy_input)
            self.flatten_size = conv_out.numel()

        # Linear projection to latent space
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, self.latent_dim),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        """Encode image observations to latent representation.
        
        Args:
            obs: Image tensor, can be LazyFrames, uint8 [0, 255], or float [0, 1].
            detach_encoder_conv: Unused, kept for API compatibility.
            detach_encoder_head: Unused, kept for API compatibility.
            
        Returns:
            Latent representation tensor of shape (batch_size, latent_dim).
        """
        # Handle LazyFrames wrapper
        if isinstance(obs, LazyFrames):
            obs = obs[:]

        # Normalize uint8 inputs to [0, 1]
        if obs.dtype == torch.uint8:
            obs = obs / 255.0

        # Convert channels-last to channels-first if needed
        if not self.channels_first:
            obs = obs.permute((0, 3, 1, 2)).contiguous()

        x = self.convs(obs)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    """Convolutional decoder for reconstructing images from latent representations.
    
    Inverse of ImageEncoder: projects latent vector back to image space using
    transposed convolutions. Output is normalized to [0, 1] via sigmoid.
    
    Architecture mirrors ImageEncoder in reverse:
    - Linear projection: latent_dim -> 3136 (64*7*7)
    - Transposed convolutions: 64->64->32->num_channels
    - Upsampling via stride > 1
    
    Args:
        obs_shape: Target image shape tuple. Can be (C, H, W) or (H, W, C).
        latent_dim: Dimension of input latent representation (default: 50).
    """

    def __init__(self, obs_shape, latent_dim=50):
        super().__init__()
        
        # Determine channel location dynamically
        self.channels_first = np.argmin(obs_shape) == 0
        self.num_channels = obs_shape[0] if self.channels_first else obs_shape[-1]
        self.img_dim = obs_shape[1]
        self.latent_dim = latent_dim

        # Feature map size after encoder conv layers: 64 * 7 * 7 = 3136
        FEATURE_MAP_SIZE = 3136
        
        # Project latent to feature map space
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, FEATURE_MAP_SIZE),
            nn.ReLU(),
        )

        # Reshape to 3D feature map (batch_size, 64, 7, 7)
        self.reshape = nn.Unflatten(1, (64, 7, 7))
        
        # Transposed convolutions: progressively upsample and reduce channels
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.num_channels, kernel_size=8, stride=4, padding=0),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        self.apply(weight_init)

    def forward(self, z):
        """Decode latent representation to image.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim).
            
        Returns:
            Reconstructed image tensor of shape (batch_size, C, H, W) with values in [0, 1].
        """
        x = self.fc(z)
        x = self.reshape(x)
        reconstructed_image = self.deconvs(x)
        return reconstructed_image
