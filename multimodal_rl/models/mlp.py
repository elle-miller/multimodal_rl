"""Multi-layer perceptron (MLP) utilities for neural network construction."""

import torch.nn as nn

_ACTIVATIONS = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


def MLP(input_dim, hidden_dims, activations, layernorm=False):
    """Build a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension.
        hidden_dims: List of hidden layer dimensions.
        activations: List of activation function names (must match hidden_dims length).
        layernorm: Whether to apply layer normalization after each linear layer.
        
    Returns:
        Sequential module containing the MLP layers.
        
    Example:
        >>> mlp = MLP(64, [128, 64], ["elu", "tanh"], layernorm=True)
        >>> x = torch.randn(32, 64)
        >>> y = mlp(x)  # Shape: (32, 64)
    """
    if not hidden_dims:
        return nn.Sequential(nn.Identity())

    if len(activations) != len(hidden_dims):
        raise ValueError(f"Number of activations ({len(activations)}) must match hidden_dims ({len(hidden_dims)})")

    layers = []
    
    # First layer: input to first hidden
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    if layernorm:
        layers.append(nn.LayerNorm(hidden_dims[0]))
    layers.append(_ACTIVATIONS[activations[0]])

    # Remaining hidden layers
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
        layers.append(_ACTIVATIONS[activations[i + 1]])

    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Simple projection network for mapping between dimensions.
    
    Architecture: Linear -> ELU -> Linear (no activation on output).
    
    Args:
        input_dim: Input dimension.
        state_dim: Output dimension.
    """

    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.ELU(),
            nn.Linear(state_dim, state_dim)
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Projected tensor of shape (batch_size, state_dim).
        """
        return self.net(x)
