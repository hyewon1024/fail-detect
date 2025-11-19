import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    """Behavioral Cloning Policy - Simple MLP"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [128, 128, 128]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        return self.network(obs)