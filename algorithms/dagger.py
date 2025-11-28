import torch
import torch.nn as nn
import torch.optim as optim
from task_utils.dataset import ExpertDataset
import random

class DAgger:
    """Standard DAgger algorithm implementation (without RND)"""
    
    def __init__(
        self,
        policy: nn.Module,
        expert,
        device: str = "cpu",
        beta_schedule: str = "linear",  # 'linear', 'exponential', or 'constant'
        beta_start: float = 1.0,
        beta_end: float = 0.0,
        policy_lr: float = 3e-4,
        batch_size: int = 256,
    ):
        self.policy = policy
        self.expert = expert
        self.device = device
        
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.current_iteration = 0
        
        self.batch_size = batch_size
        
        # Optimizer
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
        
        # Loss
        self.mse_loss = nn.MSELoss()
        
        # Dataset
        self.dataset = ExpertDataset(device=device)
    
    def get_beta(self, iteration: int, total_iterations: int):
        """
        Get mixing parameter beta for current iteration.
        beta = probability of using expert action
        """
        if self.beta_schedule == "constant":
            return self.beta_start
        
        elif self.beta_schedule == "linear":
            # Linear decay from beta_start to beta_end
            progress = iteration / max(total_iterations - 1, 1)
            return self.beta_start + (self.beta_end - self.beta_start) * progress
        
        elif self.beta_schedule == "exponential":
            # Exponential decay
            progress = iteration / max(total_iterations - 1, 1)
            return self.beta_start * (self.beta_end / self.beta_start) ** progress
        
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
    
    def collect_expert_only_data(self, env, num_steps: int):
        """
        Collect initial expert-only dataset D (Line 1 of DAgger algorithm).
        No policy is used, only expert demonstrations.
        """
        states_list = []
        actions_list = []
        
        obs, _ = env.reset()
        t = 0
        
        print("Collecting pure expert demonstrations...")
        
        while t < num_steps:
            # Only use expert
            action = self.expert.compute(obs)
            
            # Store state-action pair
            states_list.append(obs.cpu())
            actions_list.append(action.cpu())
            
            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)
            t += 1
            
            if t % 500 == 0:
                print(f"  Collected {t}/{num_steps} steps...")
            
            # Reset if done
            if terminated.any() or truncated.any():
                obs, _ = env.reset()
                self.expert.reset_idx()
        
        # Add to dataset
        if len(states_list) > 0:
            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            self.dataset.add_samples(states_tensor, actions_tensor)
        
        print(f"Finished collecting {len(states_list)} expert samples")
        return len(states_list)
    
    def collect_data(self, env, num_steps: int, beta: float):
        """
        Collect data for one DAgger iteration.
        Uses mixture of policy and expert based on beta parameter.
        
        Args:
            env: Environment
            num_steps: Number of steps to collect
            beta: Probability of using expert action (1.0 = always expert, 0.0 = always policy)
        """
        states_list = []
        actions_list = []
        
        policy_count = 0
        expert_count = 0
        
        obs, _ = env.reset()
        t = 0
        
        print(f"Collecting data with beta={beta:.3f}...")
        
        while t < num_steps:
            # Decide whether to use policy or expert based on beta
            use_expert = random.random() < beta
            
            if use_expert:
                # Execute expert action
                action = self.expert.compute(obs)
                expert_count += 1
            else:
                # Execute policy action
                with torch.no_grad():
                    action = self.policy(obs)
                policy_count += 1
            
            # Always get expert action for dataset (key part of DAgger!)
            expert_action = self.expert.compute(obs)
            
            # Store state-action pair (always expert action)
            states_list.append(obs.cpu())
            actions_list.append(expert_action.cpu())
            
            # Step environment with chosen action
            obs, _, terminated, truncated, _ = env.step(action)
            t += 1
            
            if t % 500 == 0:
                print(f"  Collected {t}/{num_steps} steps (Policy: {policy_count}, Expert: {expert_count})...")
            
            # Reset if done
            if terminated.any() or truncated.any():
                obs, _ = env.reset()
                self.expert.reset_idx()
        
        # Add to dataset
        if len(states_list) > 0:
            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            self.dataset.add_samples(states_tensor, actions_tensor)
        
        print(f"Finished collecting {len(states_list)} samples (Policy: {policy_count}, Expert: {expert_count})")
        return len(states_list)
    
    def train_policy(self, num_epochs: int = 10):
        """Train policy using BC on collected dataset"""
        dataloader = self.dataset.get_dataloader(batch_size=self.batch_size, shuffle=True)
        
        if dataloader is None:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        self.policy.train()
        for epoch in range(num_epochs):
            for states, actions in dataloader:
                # BC loss
                pred_actions = self.policy(states)
                loss = self.mse_loss(pred_actions, actions)
                
                # Update policy
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)