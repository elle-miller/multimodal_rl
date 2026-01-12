import torch
import torch.nn as nn






class DynamicsGRU(nn.Module):
    def __init__(self, 
        input_dim=128,          # Dimension of encoded state
        action_dim=1,           # CartPole has 1D action space
        gru_hidden_dim=256,
        num_layers=1,
        n_prediction_steps=5    # Number of steps to predict into future
    ):
        super().__init__()
        self.n_prediction_steps = n_prediction_steps
        
        # self.gru = nn.GRU(
        #     input_size=input_dim,
        #     hidden_size=input_dim,
        #     num_layers=num_layers,
        #     batch_first=True  # expect input shape (batch, seq, features)
        # )
        # GRU cell 
        self.gru = nn.GRUCell(
            input_size=input_dim,  # Takes encoded action as input
            hidden_size=input_dim  # Match state encoder dimension
        )

        # Predict next state encoding
        self.state_predictor = nn.Linear(input_dim, input_dim)

        # Encode actions to fit GRU input
        self.action_encoder = nn.Linear(action_dim, input_dim)

        

    def forward(self, initial_state, actions):
        """
        Args:
            initial_state: Encoded state [batch_size, state_encoder_dim]
            actions: Action sequence [batch_size, n_steps, action_dim]
        Returns:
            predicted_states: List of n_prediction_steps predicted states
        """
        
        predicted_states = []

        # Initialize hidden state with encoded initial state
        h = initial_state
        
        # Predict n steps into future
        for t in range(self.n_prediction_steps):
            # Encode action for this timestep
            encoded_action = self.action_encoder(actions[:, t])
            
            # Update GRU hidden state using action
            h = self.gru(encoded_action, h)
            
            # Predict next state
            next_state = self.state_predictor(h)
            predicted_states.append(next_state)
            
        return predicted_states
    

    
class DynamicsMLP(nn.Module):
    def __init__(
        self, 
        state_dim,          # dimension of state
        action_dim,         # dimension of action
        hidden_dim=128,     # size of hidden layers
    ):
        super().__init__()

        activation = nn.SiLU()
        activation = nn.ELU()

        
        self.net = nn.Sequential(
            # Combine state and action as input
            nn.Linear(state_dim + action_dim, 512),
            # nn.LayerNorm(512),
            activation,
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            activation,
            nn.Linear(256, state_dim),
        )
        
        # Initialize weights - can help training
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.414)
                layer.bias.data.zero_()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        next_state_delta = self.net(x)
        return state + next_state_delta  # Predict state difference
    


class ResidualMLP(nn.Module):
    def __init__(self, state_dim=128, action_dim=20, hidden_dim=256):
        super().__init__()
        
        # Residual MLP for latent dynamics
        self.net = nn.Sequential(
            # Process and merge state and action
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # SiLU/Swish activation often works well for dynamics
            
            # First residual block
            ResidualBlock(hidden_dim),
            
            # Second residual block
            ResidualBlock(hidden_dim),
            
            # Output layer - predict change in latent state
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, z, action):
        # Concatenate latent state and action
        x = torch.cat([z, action], dim=1)
        # Predict change in latent state
        delta_z = self.net(x)
        return delta_z + z

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        residual = x
        x = self.block(x)
        x = x + residual  # Skip connection
        return self.activation(x)