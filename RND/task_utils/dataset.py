import torch
from torch.utils.data import Dataset, DataLoader

class ExpertDataset(Dataset):
    """Dataset for storing state-action pairs"""
    
    def __init__(self, device: str = "cpu"):
        self.states = []
        self.actions = []
        self.expert_actions =[]
        self.device = device
    
    def add_samples(self, states, actions, expert_actions):
        """Add new samples to dataset"""
        if isinstance(states, torch.Tensor):
            self.states.append(states.cpu())
        else:
            self.states.append(torch.tensor(states, dtype=torch.float32))
        
        if isinstance(actions, torch.Tensor):
            self.actions.append(actions.cpu())
        else:
            self.actions.append(torch.tensor(actions, dtype=torch.float32))
            
        if isinstance(expert_actions, torch.Tensor):
            self.expert_actions.append(expert_actions.cpu())
        else:
            self.expert_actions.append(torch.tensor(expert_actions, dtype=torch.float32))
            

    def __len__(self):
        if len(self.states) == 0:
            return 0
        return sum(s.shape[0] for s in self.states)
    
    def __getitem__(self, idx):
        # Find which batch the index belongs to
        cumsum = 0
        for i, states_batch in enumerate(self.states):
            batch_size = states_batch.shape[0]
            if idx < cumsum + batch_size:
                local_idx = idx - cumsum
                return (
                    self.states[i][local_idx].to(self.device),
                    self.actions[i][local_idx].to(self.device)
                )
            cumsum += batch_size
        
        raise IndexError("Index out of range")
    
    def clear(self):
        """Clear all data"""
        self.states = []
        self.actions = []
        self.expert_actions =[]
    
    def get_dataloader(self, batch_size: int = 256, shuffle: bool = True):
        """Get PyTorch DataLoader"""
        if len(self) == 0:
            return None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)