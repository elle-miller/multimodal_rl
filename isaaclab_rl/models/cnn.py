import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab_rl.wrappers.frame_stack import LazyFrames



# https://github.com/denisyarats/pytorch_sac_ae/blob/master/sac_ae.py
#  mhairi - https://github.com/uoe-agents/MVD/blob/main/utils.py
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class ImageEncoder(nn.Module):
    """Convolutional encoder for image-based observations.

    You (2024) construact an image observation by stacking 3 frames,
    where each frame is 84x84x3. We divide each pixel by 255 and scale
    it down to [0,1]. Before we feed images into the encoders, we follow
    Yarats 2021 data augmentation by random shift [-4, 4]

    obs_shape (C, W, H)

    nn.Conv2d(in_channels, out_channels, kernel_size)
    """

    def __init__(self, obs_shape, latent_dim=50, num_layers=4, num_filters=32):
        super().__init__()


        self.num_layers = num_layers
        self.num_filters = num_filters
        self.latent_dim = latent_dim

        # dynamically determine where the channels location
        self.channels_first = True if np.argmin(obs_shape) == 0 else False
        self.num_channels = obs_shape[0] if self.channels_first else obs_shape[-1]
        self.img_dim = obs_shape[1]

        print("initialising cnn with ", num_layers, "and ", latent_dim, "feature dim", "num channels", self.num_channels, self.img_dim)

        # First layer: 8x8 kernel focuses on broad features for downsampling
        # Second layer: reduce features, but increase channels for more features
        self.convs = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=8, stride=4, padding=0),  # Output: 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # Output: 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # Output: 64 x 7 x 7
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # Dynamically compute the flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_channels, self.img_dim, self.img_dim)
            conv_out = self.convs(dummy_input)
            self.flatten_size = conv_out.numel()

        # Calculate the input size of the linear layer based on input resolution (e.g., 84x84)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, self.latent_dim),
            nn.ReLU(),
            # nn.LayerNorm(self.latent_dim)
        )

        self.apply(weight_init)

    def forward(self, obs, detach_encoder_conv=False, detach_encoder_head=False):
        if isinstance(obs, LazyFrames):
            obs = obs[:]

        # convert to torch32 if uint8
        if obs.dtype is torch.uint8:
            obs = obs / 255.0

        if not self.channels_first:
            # Permuting makes the tensor non-contiguous(?)
            obs = obs.permute((0, 3, 1, 2)).contiguous()

        x = obs
        x = self.convs(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    """
    Reverses this:

    ImageEncoder(
    (convs): ModuleList(
        (0): Conv2d(9, 32, kernel_size=(5, 5), stride=(2, 2))
        (1-3): 3 x Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    )
    (head): Sequential(
        (0): Linear(in_features=36992, out_features=50, bias=True)
        (1): LayerNorm((50,), eps=1e-05, elementwise_affine=True)
    )
    )

    """

    def __init__(self, obs_shape, latent_dim=50):
        super().__init__()

        # dynamically determine where the channels location
        self.channels_first = True if np.argmin(obs_shape) == 0 else False
        self.num_channels = obs_shape[0] if self.channels_first else obs_shape[-1]
        self.img_dim = obs_shape[1]
        self.latent_dim = latent_dim

        # Fully connected layer to map latent vector back to feature map
        self.fc = nn.Sequential(
            nn.Linear(
                self.latent_dim, 3136
            ),  # This should match the flattened output of your encoder (64 * 7 * 7 = 3136)
            nn.ReLU(),
        )

        # Reshape the output to a 3D tensor (batch_size, 64, 7, 7) to pass to transposed convolutions
        self.reshape = nn.Unflatten(1, (64, 7, 7))
        # Transposed convolutions (deconvolutions) to upsample the feature map to the image size
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),  # Output: 64 x 7 x 7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),  # Output: 32 x 9 x 9
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, self.num_channels, kernel_size=8, stride=4, padding=0
            ),  # Output: num_channels x 20 x 20
            nn.Sigmoid(),  # Use sigmoid for pixel values in the range [0, 1] (assuming RGB images)
        )

        self.apply(weight_init)

    def forward(self, z):
        # Pass through the fully connected layer
        x = self.fc(z)

        # Reshape into a 3D tensor (batch_size, 64, 7, 7)
        x = self.reshape(x)

        # Pass through the deconvolution layers
        reconstructed_image = self.deconvs(x)

        # bad todo fix to check dims
        # reconstructed_image = reconstructed_image.permute((0, 2, 3, 1))

        return reconstructed_image
