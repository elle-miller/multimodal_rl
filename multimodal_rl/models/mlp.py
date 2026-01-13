import torch.nn as nn

activations = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


def MLP(inputs, hiddens, hidden_activations, layernorm=False):
    layers = []

    if hiddens == []:
        return nn.Sequential(nn.Identity())

    # First hidden layer: from inputs to the first hidden layer
    layers.append(nn.Linear(inputs, hiddens[0]))
    if layernorm:
        layers.append(nn.LayerNorm(hiddens[0])),
    layers.append(activations[hidden_activations[0]])  # Add activation

    # Hidden layers: loop over hidden layers
    for i in range(len(hiddens) - 1):
        layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(hiddens[i + 1])),
        layers.append(activations[hidden_activations[i + 1]])  # Add activation

    return nn.Sequential(*layers)


class Projector(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, state_dim), nn.ELU(), nn.Linear(state_dim, state_dim))

    def forward(self, x):
        return self.net(x)
