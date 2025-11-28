import torch
from skrl.utils import set_seed

from env import set_env_dataCollection
from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy

from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger import RNDDAgger
import os

# Seed
set_seed(42)

# Hyperparameters
K_ITERATIONS = 200
STEPS_PER_ITERATION = 2
INITIAL_EXPERT_STEPS = 2000 #2000
EVAL_STEPS = 100  # Steps for policy evaluation

LAMBDA_THRESHOLD = 1
MIN_DEMO_TIME = 20
HISTORIC_CONTEXT_LENGTH = 2  # H: 0, 1, 2, ... (number of past observations)


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
policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
f_targ, f_pred = create_rnd_networks(obs_dim, output_dim = 128, device=device)

policy.load_state_dict(torch.load('/AILAB-summer-school-2025/RND/checkpoints/rnd_lambda5_minDemo50_H5/policy_iter_80.pt', map_location=device))
f_pred.load_state_dict(torch.load('/AILAB-summer-school-2025/checkpoints/dagger_iter_15.pt', map_location=device)["f_pred_state_dict"])
f_targ.load_state_dict(torch.load('/AILAB-summer-school-2025/checkpoints/dagger_iter_15.pt', map_location=device)["f_targ_state_dict"])

# Initialize RND-DAgger
dagger = RNDDAgger(
    policy=policy,
    f_targ=f_targ,
    f_pred=f_pred,
    expert=expert,
    device=device,
    lambda_threshold=LAMBDA_THRESHOLD,
    min_demo_time=MIN_DEMO_TIME,
    policy_lr=3e-3,
    rnd_lr=1e-3,
    batch_size=32
)

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

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
for epoch in range(1000):  # More epochs for initial training
    policy_loss = dagger.train_policy(num_epochs=1)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/50 - Policy loss: {policy_loss:.4f}")

print(f"Initial policy training complete. Final loss: {policy_loss:.4f}")

# Save initial policy
torch.save(policy.state_dict(), 'checkpoints/policy_pi0.pt')
print("Saved initial policy π₀")

print("\n" + "="*60)
print("STEP 3: Starting RND-DAgger iterations")
print("="*60)

# Training loop (DAgger iterations)
for iteration in range(K_ITERATIONS):
    print(f"\n=== Iteration {iteration + 1}/{K_ITERATIONS} ===")
    
    # Collect data (mixture of policy and expert)
    print("Collecting data with policy + expert intervention...")
    num_samples = dagger.collect_data(env, STEPS_PER_ITERATION)
    print(f"Collected {num_samples} samples. Total dataset size: {len(dagger.dataset)}")
    print(f"Context switches so far: {dagger.nswitch}")
    
    # Train policy
    print("Training policy...")
    policy_loss = dagger.train_policy(num_epochs=100)
    print(f"Policy loss: {policy_loss:.4f}")
    
    # Train RND
    print("Training RND predictor...")
    rnd_loss = dagger.train_rnd(num_epochs=50)
    print(f"RND loss: {rnd_loss:.4f}")
    
    # Save checkpoint
    if (iteration + 1) % 5 == 0:
        torch.save({
            'iteration': iteration + 1,
            'policy_state_dict': policy.state_dict(),
            'f_pred_state_dict': f_pred.state_dict(),
            'optimizer_state_dict': dagger.policy_optimizer.state_dict(),
            'dataset_size': len(dagger.dataset),
            'nswitch': dagger.nswitch,
        }, f'checkpoints/dagger_iter_{iteration + 1}.pt')
        print(f"Saved checkpoint at iteration {iteration + 1}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Total context switches: {dagger.nswitch}")
print(f"Final dataset size: {len(dagger.dataset)}")

# Save final model
torch.save(policy.state_dict(), 'checkpoints/final_policy.pt')
print("Saved final policy!")