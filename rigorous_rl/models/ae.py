import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, input_size, hidden_size=512, feature_dim=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, feature_dim), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, input_size), nn.Tanh()
        )
        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

    def reconstruct(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE(nn.Module):

    def __init__(self, input_size=784, hidden_size=400, feature_dim=200):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, feature_dim), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )
        self.enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # latent mean and variance
        self.mean_layer = nn.Linear(feature_dim, 2)
        self.logvar_layer = nn.Linear(feature_dim, 2)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def reconstruct(self, x):
        z = self.forward(x)
        x_hat = self.decode(z)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        return z
