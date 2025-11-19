import torch
import torch.nn as nn
import torch.optim as optim
import random
from task_utils.dataset import ExpertDataset

class RNDDAgger:
    """RND-DAgger algorithm implementation"""
    
    def __init__(
        self,
        policy: nn.Module,
        f_targ: nn.Module,
        f_pred: nn.Module,
        expert,
        device: str = "cpu",
        lambda_threshold: float = 2.0,
        min_demo_time: int = 5,
        historic_context_length: int = 0,
        policy_lr: float = 3e-4,
        rnd_lr: float = 1e-4,
        batch_size: int = 256,
        logger= None,
    ):
        self.policy = policy
        self.f_targ = f_targ
        self.f_pred = f_pred
        self.expert = expert
        self.device = device
        
        self.lambda_threshold = lambda_threshold
        self.min_demo_time = min_demo_time
        self.historic_context_length = historic_context_length
        self.batch_size = batch_size
        
        # Optimizers
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
        self.rnd_optimizer = optim.Adam(f_pred.parameters(), lr=rnd_lr)
        
        # Loss
        self.mse_loss = nn.MSELoss()
        
        # Dataset
        self.dataset = ExpertDataset(device=device)
        
        # Tracking
        self.nswitch = 0
        self.w_counter = self.min_demo_time + 1
        self.logger = logger
        self.global_step = 0
        self.episode_count = 1

        self.m_previous = 0.0
        
        # Historic context buffer (for each environment)
        self.obs_history = None

        self.beta_schedule: str = "exponential"  # 'linear', 'exponential', or 'constant'
        self.beta_start: float = 1.0
        self.beta_end: float = 0.0

    
    def _init_history_buffer(self, num_envs: int, obs_dim: int):
        """Initialize observation history buffer."""
        if self.historic_context_length > 0:
            # Buffer shape: [num_envs, historic_context_length, obs_dim]
            self.obs_history = torch.zeros(
                num_envs, 
                self.historic_context_length, 
                obs_dim, 
                device=self.device
            )

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
            return self.beta_start * 0.99 ** iteration
        
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
    def _update_history(self, obs):
        """Update observation history with new observation."""
        if self.historic_context_length > 0:
            # Shift history and add new observation
            self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
            self.obs_history[:, -1, :] = obs
    
    def _get_context_obs(self, obs):
        """Get observation with historical context."""
        if self.historic_context_length == 0:
            return obs
        else:
            # Flatten history: [num_envs, historic_context_length * obs_dim]
            history_flat = self.obs_history.reshape(obs.shape[0], -1)
            # Concatenate current obs with history
            return torch.cat([obs, history_flat], dim=1)
    
    def compute_ood_measure(self, obs):
        """Compute OOD measure using RND with historic context."""
        context_obs = self._get_context_obs(obs)
        
        with torch.no_grad():
            pred = self.f_pred(context_obs)
            targ = self.f_targ(context_obs)
            m = torch.norm(targ - pred, dim=-1) ** 2
        return m
    
    def collect_expert_only_data(self, env, num_steps: int):
        """Collect initial expert-only dataset D."""
        states_list = []
        actions_list = []
        env.unwrapped.reset()

        obs, _ = env.reset()
        self.expert.reset_idx()
        
        # Initialize history buffer
        if self.obs_history is None:
            self._init_history_buffer(obs.shape[0], obs.shape[1])
        
        t = 0
        rewards = 0

        print("Collecting pure expert demonstrations...")
        
        while t < num_steps:
            # Update history
            self._update_history(obs)
            
            # Only use expert
            action = self.expert.compute(obs)
            
            # Store state-action pair
            states_list.append(obs.cpu())
            actions_list.append(action.cpu())
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            t += 1
            rewards += reward

            if t % 500 == 0:
                print(f"  Collected {t}/{num_steps} steps...")
            
            # Reset if done
            if terminated.any() or truncated.any():
                obs, _ = env.reset()
                self.expert.reset_idx()
                # Reset history for done environments
                if self.historic_context_length > 0:
                    done_mask = terminated | truncated
                    done_indices = torch.where(done_mask)[0]
                    if len(done_indices) > 0:
                        self.obs_history[done_indices] = 0.0
        
        # Add to dataset
        if len(states_list) > 0:
            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            self.dataset.add_samples(states_tensor, actions_tensor)
        
        print(f"Finished collecting {len(states_list)} expert samples, rewards {rewards}")
        return len(states_list)
    
    def evaluate_policy(self, env, num_steps: int = 1000):
        """
        Evaluate policy without expert intervention.
        Returns: (total_reward, num_episodes)
        """
        print("Evaluating policy (no expert intervention)...")
        env.unwrapped.reset()

        obs, _ = env.reset()
        
        # Reset history buffer
        if self.historic_context_length > 0:
            self.obs_history.zero_()
        
        total_reward = 0.0
        num_episodes = 0
        t = 0
        
        self.policy.eval()
        
        while t < num_steps:
            # Update history
            self._update_history(obs)
            
            # Use policy only (no expert)
            with torch.no_grad():
                action = self.policy(obs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward.sum().item()
            t += 1
            
            # Count episodes
            done = terminated | truncated
            if done.any():
                num_episodes += done.sum().item()
                
                # Reset history for done environments
                if self.historic_context_length > 0:
                    done_indices = torch.where(done)[0]
                    if len(done_indices) > 0:
                        self.obs_history[done_indices] = 0.0
            
            if t % 200 == 0:
                print(f"  Evaluated {t}/{num_steps} steps...")
        
        self.policy.train()
        
        avg_reward = total_reward / num_steps
        print(f"Evaluation complete: Avg Reward = {avg_reward:.4f}, Episodes = {num_episodes}")
        
        return avg_reward, num_episodes
    
    def collect_data(self, env, num_steps: int, beta: float):
        """Collect data for one DAgger iteration with RND-based intervention."""
        env.unwrapped.reset()
        states_list = []
        actions_list = []

        obs, _ = env.reset()
        self.w_counter = self.min_demo_time + 1
        self.expert.reset_idx()
        self.nswitch = 0
        
        # Initialize history buffer if needed
        if self.obs_history is None:
            self._init_history_buffer(obs.shape[0], obs.shape[1])
        
        t = 0
        while t < num_steps:
            self.global_step += 1
            self._update_history(obs)

            m = self.compute_ood_measure(obs)
            m_scalar = m.mean().item()
            self.logger.log(
                global_step=self.global_step,
                lambda_val=self.lambda_threshold,
                ood=m_scalar,
                episode=self.episode_count
            )

            # Decide whether to use policy or expert based on beta
            use_expert = random.random() < beta
            
            if use_expert:
                # Execute expert action
                action = self.expert.compute(obs)
            else:
                # Execute policy action
                with torch.no_grad():
                    action = self.policy(obs)
            
            # Decide who controls the agent
            if m.mean() > self.lambda_threshold:
                # Expert control
                self.nswitch += 1 
                expert_action = self.expert.compute(obs)
                states_list.append(obs.cpu())
                actions_list.append(expert_action.cpu())
            else:     
                # Learner Policy 
                states_list.append(obs.cpu())
                actions_list.append(action.cpu())
                                        
            # Step environment with chosen action
            #print("current m.mean(): ", m.mean())
            obs, _, terminated, truncated, _ = env.step(action)
            t += 1

        # while t < num_steps:
        #     # Update history
        #     self._update_history(obs)
            
        #     # Compute OOD measure
        #     m = self.compute_ood_measure(obs)
            
        #     # Decide who controls the agent
        #     if m.mean() > self.lambda_threshold or self.w_counter < self.min_demo_time:
        #         # Expert control
        #         if m.mean() < self.lambda_threshold:
        #             self.w_counter += 1
        #         else:
        #             self.w_counter = 0

        #         action = self.expert.compute(obs)
        #         states_list.append(obs.cpu())
        #         actions_list.append(action.cpu())

        #         if self.m_previous < self.lambda_threshold:
        #             self.nswitch += 1
        #         self.m_previous = m.mean()

        #     else:
        #         # Policy control
        #         with torch.no_grad():
        #             action = self.policy(obs)

        #     # Step environment
        #     obs, _, terminated, truncated, _ = env.step(action)
        #     t += 1
            
            # Reset if done
            if terminated.any() or truncated.any():
                obs, _ = env.reset()
                self.w_counter = self.min_demo_time + 1
                self.expert.sm_state[0] = 0
                
                # Reset history for done environments
                if self.historic_context_length > 0:
                    done_mask = terminated | truncated
                    done_indices = torch.where(done_mask)[0]
                    if len(done_indices) > 0:
                        self.obs_history[done_indices] = 0.0
                self.episode_count += 1     
                self.logger.log_episode_end(self.episode_count)   
        print("current switch number from learner to expert", self.nswitch)
        # Add to dataset
        if len(states_list) > 0:
            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            self.dataset.add_samples(states_tensor, actions_tensor)
        
        return len(states_list)
    
    def train_policy(self, num_epochs: int = 10):
        """Train policy using BC on collected dataset."""
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
    
    def train_rnd(self, num_epochs: int = 5):
        """Train RND predictor network."""
        dataloader = self.dataset.get_dataloader(batch_size=self.batch_size, shuffle=True)
        
        if dataloader is None:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        self.f_pred.train()
        for epoch in range(num_epochs):
            for states, _ in dataloader:
                # For RND, we need context observations
                # Since dataset only has current obs, we'll use zero-padded history
                if self.historic_context_length > 0:
                    batch_size = states.shape[0]
                    obs_dim = states.shape[1]
                    # # # Zero-pad history for training
                    history_padding = torch.zeros(
                        batch_size, 
                        self.historic_context_length * obs_dim,
                        device=states.device
                    )
                    # flattened_history = self.obs_history.reshape(1, self.historic_context_length * obs_dim)
                    context_states = torch.cat([states, history_padding], dim=1)
                else:
                    context_states = states
                
                # RND loss
                pred = self.f_pred(context_states)
                with torch.no_grad():
                    targ = self.f_targ(context_states)
                loss = self.mse_loss(pred, targ)
                
                # Update RND predictor
                self.rnd_optimizer.zero_grad()
                loss.backward()
                self.rnd_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)