import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset
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
        calib_split_ratio: float = 0.5,
        alpha: float = 0.1,
        alg = None, 
    ):
        self.policy = policy
        self.f_targ = f_targ
        self.f_pred = f_pred
        self.expert = expert
        self.device = device
        self.alg = alg

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

        # Split conformal params
        self.calib_split_ratio = calib_split_ratio
        self.alpha = alpha
        self.train_states = None
        self.calib_states = None

        self.beta_schedule: str = "exponential"  # 'linear', 'exponential', or 'constant'
        self.beta_start: float = 1.0
        self.beta_end: float = 0.0
        self.adaptive_lambda = None 

    
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
    
    def quantile(self, alpha: float = None):
        """
        Compute CP-style quantile using only calibration split.
        q_level = (n + 1) * (1 - alpha) / n
        """
        if alpha is None:
            alpha = self.alpha

        if self.calib_states is None or self.calib_states.numel() == 0:
            print("No calibration data available, using existing lambda threshold")
            return self.lambda_threshold, self.lambda_threshold, self.lambda_threshold

        calib_states = self.calib_states
        n = calib_states.shape[0]
        q_level = (n + 1) * (1 - alpha) / max(n, 1)
        q_level = min(q_level, 1.0)

        was_pred_training = self.f_pred.training
        was_targ_training = self.f_targ.training
        self.f_pred.eval()
        self.f_targ.eval()

        # Compute scores in chunks, moving only batches to device to avoid OOM.
        scores = []
        calib_loader = DataLoader(TensorDataset(calib_states), batch_size=8192, shuffle=False)
        with torch.no_grad():
            for (batch_states,) in calib_loader:
                batch_states = batch_states.to(self.device, non_blocking=True)
                pred = self.f_pred(batch_states)
                targ = self.f_targ(batch_states)
                batch_scores = torch.norm(targ - pred, dim=-1) ** 2
                scores.append(batch_scores.cpu())

        if not scores:
            print("No calibration data available, using existing lambda threshold")
            return self.lambda_threshold, self.lambda_threshold, self.lambda_threshold

        scores = torch.cat(scores, dim=0)

        if was_pred_training:
            self.f_pred.train()
        if was_targ_training:
            self.f_targ.train()

        lambda_value = torch.quantile(scores, q_level).item()
        min_lambda = torch.min(scores).item()
        max_lambda = torch.max(scores).item()
        self.lambda_threshold = lambda_value
        print(f"Calibrated lambda value is : {lambda_value}")
        return lambda_value, min_lambda, max_lambda
       
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
            self.dataset.add_samples(states_tensor, actions_tensor, actions_tensor)

            # Keep the aggregated dataset on CPU to avoid blowing up GPU memory.
            states_all = torch.cat(self.dataset.states, dim=0).cpu()
            total = states_all.shape[0]
            perm = torch.randperm(total)
            train_size = int(total * (1 - self.calib_split_ratio))
            train_idx = perm[:train_size]
            calib_idx = perm[train_size:]
            self.train_states = states_all[train_idx] if train_size > 0 else None
            self.calib_states = states_all[calib_idx] if calib_idx.numel() > 0 else None
        
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
    
    def collect_data(self, env, num_steps: int, beta: float, alpha_=None):
        """Collect data for one DAgger iteration with RND-based intervention."""
        env.unwrapped.reset()
        states_list = []
        actions_list = []
        expert_actions_list =[]
        obs, _ = env.reset()
        self.w_counter = self.min_demo_time + 1
        self.expert.reset_idx()
        self.nswitch = 0
        # Initialize history buffer if needed
        if self.obs_history is None:
            self._init_history_buffer(obs.shape[0], obs.shape[1])
            
        adaptive_lambda, _, _ = self.quantile(alpha_)
        t = 0

        while t < num_steps:
            self.global_step += 1
            self._update_history(obs)
            m = self.compute_ood_measure(obs)
            m_scalar = m.mean().item()
            if self.logger is not None:
                self.logger.log(
                    global_step=self.global_step,
                    lambda_val=max(adaptive_lambda, 0.01),
                    ood=m_scalar,
                    episode=self.episode_count
                )
            # Decide whether to use policy or expert based on RND
            if self.alg =="pure_dagger":
                use_expert = random.random() < beta
            else: 
                use_expert = m.mean() > max(adaptive_lambda, 0.01) 

            if use_expert:
                expert_action = self.expert.compute(obs)
                self.nswitch += 1
                states_list.append(obs.cpu())
                actions_list.append(expert_action.cpu())
                obs, _, terminated, truncated, _ = env.step(expert_action)
                expert_actions_list.append(expert_action.cpu()) # ADD
            else:
                action = self.policy(obs)
                expert_action = self.expert.compute(obs)
                states_list.append(obs.cpu())
                actions_list.append(expert_action.cpu())
                obs, _, terminated, truncated, _ = env.step(action)
                expert_actions_list.append(expert_action.cpu()) # ADD

            t += 1
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
            if self.logger is not None:   
                self.logger.log_episode_end(self.episode_count)

        print("current switch number from learner to expert", self.nswitch)
        # Add to dataset
        if len(states_list) > 0:
            states_tensor = torch.cat(states_list, dim=0)
            actions_tensor = torch.cat(actions_list, dim=0)
            expert_actions_tensor = torch.cat(expert_actions_list, dim=0)
            self.dataset.add_samples(states_tensor, actions_tensor, expert_actions_tensor)

            # Keep aggregated tensors on CPU; move to GPU only per-batch during training.
            states_all = torch.cat(self.dataset.states, dim=0).cpu()
            total = states_all.shape[0]
            perm = torch.randperm(total)
            train_size = int(total * (1 - self.calib_split_ratio))
            train_idx = perm[:train_size]
            calib_idx = perm[train_size:]
            self.train_states = states_all[train_idx] if train_size > 0 else None
            self.calib_states = states_all[calib_idx] if calib_idx.numel() > 0 else None
        
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


    #-------------
    def train_rnd_balanced(
        self,
        num_epochs: int = 300,
        n_proj: int = 8,
        eps: float = 0.1,
        seed: int = 1,
    ):

        torch.manual_seed(seed)
        device = self.device
        self.f_pred.train()

        for p in self.f_targ.parameters():
            p.requires_grad_(False)

        if self.train_states is None:
            return 0.0

        dataset = TensorDataset(self.train_states)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0.0
        total_batches = 0

        for epoch in range(num_epochs):
            for (states,) in dataloader:
                states = states.to(device, non_blocking=True)

                N, D = states.shape
                idx = torch.randint(0, N, (self.batch_size,), device=device)
                z = states[idx]                      # (B, D)

                # --- Base RND loss ---
                with torch.no_grad():
                    t = self.f_targ(z)
                p = self.f_pred(z)

                E = (p - t).pow(2).sum(dim=-1)       # (B,)
                loss_rnd = E.mean()

                v = torch.randn(n_proj, D, device=device)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

                # 각 방향마다 perturbation 에너지 변화 측정
                deltas_mean = []
                for i in range(n_proj):
                    delta = eps * v[i].unsqueeze(0)  # (1, D)
                    z_pert = z + delta

                    with torch.no_grad():
                        targ_pert = self.f_targ(z_pert)
                    pred_pert = self.f_pred(z_pert)

                    E_pert = (pred_pert - targ_pert).pow(2).sum(dim=-1)  # scalar
                    deltas_mean.append((E_pert - E).abs().mean())

                deltas_mean = torch.stack(deltas_mean)
                # log scale에서 variance 줄이기
                log_deltas = torch.log(deltas_mean + 1e-8)
                loss_bal = log_deltas.var()

                loss = loss_rnd + loss_bal

                self.rnd_optimizer.zero_grad()
                loss.backward()
                self.rnd_optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            # print(
            #     f"[epoch {epoch+1:4d}] "
            #     f"RND={loss_rnd.item():.6f} Bal={loss_bal.item():.6f}"
            # )

        avg_loss = total_loss / max(total_batches, 1)
        self.quantile()
        return avg_loss
    
    #------------
    def train_rnd(self, num_epochs: int = 5):
        """Train RND predictor network."""
        if self.train_states is None:
            return 0.0

        dataset = TensorDataset(self.train_states)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_loss = 0.0
        num_batches = 0
        
        self.f_pred.train()
        for epoch in range(num_epochs):
            for (states,) in dataloader:
                states = states.to(self.device, non_blocking=True)
                # RND loss
                pred = self.f_pred(states)
                with torch.no_grad():
                    targ = self.f_targ(states)
                loss = self.mse_loss(pred, targ)
                
                # Update RND predictor
                self.rnd_optimizer.zero_grad()
                loss.backward()
                self.rnd_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.quantile()
        return avg_loss