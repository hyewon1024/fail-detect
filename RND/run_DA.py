import torch
from skrl.utils import set_seed

from env import set_env_dataCollection
from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy

from models.bc_policy import BCPolicy
from algorithms.dagger import DAgger
import os

# Seed
set_seed(42)

# Hyperparameters
K_ITERATIONS = 60
STEPS_PER_ITERATION = 2000
INITIAL_EXPERT_STEPS = 5000

# Beta schedule options: 'linear', 'exponential', 'constant'
BETA_SCHEDULE = "linear"
BETA_START = 1.0  # Start with 100% expert
BETA_END = 0.1    # End with 10% expert

# Get dimensions from environment
obs_dim = env.observation_space.shape[0]  # 25
action_dim = env.action_space.shape[0]    # 8

print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

# Initialize expert policy
expert = ExpertPolicy(
    dt=set_env_dataCollection.env_cfg.sim.dt * set_env_dataCollection.env_cfg.decimation,
    num_envs=env.unwrapped.num_envs,
    device=device,
    env=env.unwrapped
)

# Initialize networks
policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128]).to(device)

# Initialize DAgger
dagger = DAgger(
    policy=policy,
    expert=expert,
    device=device,
    beta_schedule=BETA_SCHEDULE,
    beta_start=BETA_START,
    beta_end=BETA_END,
    policy_lr=3e-4,
    batch_size=256
)

# Create checkpoints directory
os.makedirs('checkpoints_DA2', exist_ok=True)

print("\n" + "="*60)
print("STEP 1: Collecting initial expert dataset D")
print("="*60)

# Collect initial expert demonstrations
print(f"Collecting {INITIAL_EXPERT_STEPS} steps of expert data...")
num_samples = dagger.collect_expert_only_data(env, INITIAL_EXPERT_STEPS)
print(f"Collected {num_samples} expert samples")
print(f"Initial dataset size: {len(dagger.dataset)}")

print("\n" + "="*60)
print("STEP 2: Training initial policy π₀ on expert dataset D")
print("="*60)

# Train initial policy π₀ on expert data
print("Pre-training policy on expert dataset...")
for epoch in range(50):
    policy_loss = dagger.train_policy(num_epochs=1)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/50 - Policy loss: {policy_loss:.4f}")

print(f"Initial policy training complete. Final loss: {policy_loss:.4f}")

# Save initial policy
torch.save(policy.state_dict(), 'checkpoints_DA2/policy_pi0.pt')
print("Saved initial policy π₀")

print("\n" + "="*60)
print("STEP 3: Starting DAgger iterations")
print("="*60)

# Training loop (DAgger iterations)
for iteration in range(K_ITERATIONS):
    print(f"\n=== Iteration {iteration + 1}/{K_ITERATIONS} ===")
    
    # Compute beta for this iteration
    beta = dagger.get_beta(iteration, K_ITERATIONS)
    print(f"Beta (expert probability): {beta:.3f}")
    
    # Collect data (mixture of policy and expert)
    print("Collecting data with policy + expert mixture...")
    num_samples = dagger.collect_data(env, STEPS_PER_ITERATION, beta=beta)
    print(f"Collected {num_samples} samples. Total dataset size: {len(dagger.dataset)}")
    
    # Train policy
    print("Training policy...")
    policy_loss = dagger.train_policy(num_epochs=10)
    print(f"Policy loss: {policy_loss:.4f}")
    
    # Save checkpoint
    if (iteration + 1) % 5 == 0:
        torch.save({
            'iteration': iteration + 1,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': dagger.policy_optimizer.state_dict(),
            'dataset_size': len(dagger.dataset),
            'beta': beta,
        }, f'checkpoints_DA2/dagger_iter_{iteration + 1}.pt')
        print(f"Saved checkpoint at iteration {iteration + 1}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Final dataset size: {len(dagger.dataset)}")
print(f"Final beta: {dagger.get_beta(K_ITERATIONS - 1, K_ITERATIONS):.3f}")

# Save final model
torch.save(policy.state_dict(), 'checkpoints_DA2/final_policy_dagger.pt')
print("Saved final policy!")