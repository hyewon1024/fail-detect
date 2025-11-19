import torch
import torch.nn as nn

class RNDNetwork(nn.Module):
    """Random Network Distillation - for OOD detection"""
    
    def __init__(self, obs_dim: int, output_dim: int = 32):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, obs):
        return self.network(obs)


class RNDPredictor(nn.Module):
    """Predictor network for RND (trainable)"""
    
    def __init__(self, obs_dim: int, output_dim: int = 32):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        return self.network(obs)


def create_rnd_networks(obs_dim: int, historic_context_length: int = 0, output_dim: int = 32, device: str = "cpu", freeze: bool = True):
    """
    Create RND target (fixed) and predictor (trainable) networks.
    
    Args:
        obs_dim: Observation dimension
        historic_context_length: Number of past observations to include
        output_dim: RND embedding dimension
        device: Device to create networks on
    """
    # Calculate total input dimension
    # If H=0: input_dim = obs_dim
    # If H=1: input_dim = obs_dim + obs_dim = 2 * obs_dim
    # If H=2: input_dim = obs_dim + 2 * obs_dim = 3 * obs_dim
    total_input_dim = obs_dim * (1 + historic_context_length)
    
    print(f"Creating RND networks with input_dim={total_input_dim} (obs_dim={obs_dim}, H={historic_context_length})")
    
    # Target network - frozen with Xavier initialization
    f_targ = RNDNetwork(total_input_dim, output_dim).to(device)
    
    # Apply Xavier initialization only to weight matrices
    for module in f_targ.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    # Freeze all parameters
    for param in f_targ.parameters():
        param.requires_grad = False
    f_targ.eval()
    
    # Predictor network - trainable
    f_pred = RNDPredictor(total_input_dim, output_dim).to(device)
    if freeze:
        for name, param in f_targ.named_parameters():
            if not name.startswith("network.4"):
                param.requires_grad = False

    return f_targ, f_pred