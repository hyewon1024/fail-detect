import torch
from skrl.utils import set_seed

from env import set_env_dataCollection
from task_utils.setting_config import device, env
from models.bc_policy import BCPolicy
import os

# Seed
set_seed(42)

print("="*60)
print("Trained Policy Visualization")
print("="*60)

# Get dimensions
obs_dim = env.observation_space.shape[0]  # 25
action_dim = env.action_space.shape[0]    # 8

print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

# Initialize policy
policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)

# Load trained policy
checkpoint_path = "/AILAB-summer-school-2025/RND/checkpoints/[rnd_dagger]rnd_iter200_balance_True_epoch_10_alpha_0.7_minDemo70_H0_FTrue/policy_iter_90.pt"

if not os.path.exists(checkpoint_path):
    print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
    print("Available checkpoints:")
    if os.path.exists('checkpoints'):
        for f in os.listdir('checkpoints'):
            print(f"  - checkpoints/{f}")
    else:
        print("No checkpoints directory found!")
    exit(1)

print(f"\nLoading policy from: {checkpoint_path}")


policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
policy.eval()  # Set to evaluation mode

print("Policy loaded successfully!\n")

# Reset environment
obs, _ = env.reset()

print(f"Running trained policy...")
print(f"Number of environments: {env.unwrapped.num_envs}")
print(f"Press Ctrl+C to stop\n")

step_count = 0
episode_count = 0
success_count = 0
total_reward = 0.0

try:
    while True:
        # Get action from trained policy
        with torch.no_grad():
            actions = policy(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        step_count += 1
        total_reward += reward.sum().item()
        
        # Print status every 100 steps
        if step_count % 100 == 0:
            avg_reward = total_reward / step_count / env.unwrapped.num_envs
            print(f"Step: {step_count}, Episodes: {episode_count}, Successes: {success_count}, Avg Reward: {avg_reward:.3f}")
        
        # Check if any environment is done
        done = terminated | truncated
        if done.any():
            episode_count += done.sum().item()
            
            # Count successes (check if object is in bin)
            for i in range(env.unwrapped.num_envs):
                if done[i]:
                    obj_pos = obs[i][18:21]  # Object position in robot frame
                    bin_pos = obs[i][21:24]  # Bin position in robot frame
                    
                    # Check if object is in bin
                    xy_dist = torch.norm(obj_pos[:2] - bin_pos[:2])
                    if xy_dist < 0.15 and obj_pos[2] < 0.1:
                        success_count += 1
                        print(f"  ✓ Success in env {i}!")
            
            # Print success rate
            if episode_count > 0:
                success_rate = (success_count / episode_count) * 100
                print(f"\n>>> Current Success Rate: {success_rate:.1f}% ({success_count}/{episode_count})\n")

except KeyboardInterrupt:
    print("\n" + "="*60)
    print("Visualization stopped by user")
    print("="*60)
    print(f"Total steps: {step_count}")
    print(f"Total episodes: {episode_count}")
    print(f"Total successes: {success_count}")
    if episode_count > 0:
        success_rate = (success_count / episode_count) * 100
        print(f"Final success rate: {success_rate:.1f}%")
    avg_reward = total_reward / step_count / env.unwrapped.num_envs if step_count > 0 else 0
    print(f"Average reward: {avg_reward:.3f}")
    print("="*60)

env.close()