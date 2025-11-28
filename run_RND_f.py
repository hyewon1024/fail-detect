import torch
from skrl.utils import set_seed

from env import set_env_dataCollection
from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy

from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger_adaptive import RNDDAgger
import os
import json
import re
from collections.abc import Sequence
# from task_utils.ood_logger import OODLogger
import numpy as np 

# # Hyperparameter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_folder", type=str, required=True)
parser.add_argument("--iter_num", type=int, required=True)

args, unknown = parser.parse_known_args()

print("[DEBUG] argparse args:", args)
print("[DEBUG] unknown args:", unknown)

if args.ckpt_folder is None and "--ckpt_folder" in unknown:
    idx = unknown.index("--ckpt_folder")
    args.ckpt_folder = unknown[idx + 1]

if args.iter_num is None and "--iter_num" in unknown:
    idx = unknown.index("--iter_num")
    args.iter_num = int(unknown[idx + 1])

folder_path = args.ckpt_folder
iter_num = args.iter_num
folder_name = os.path.basename(folder_path)

STEPS_PER_ITERATION = 2000
INITIAL_EXPERT_STEPS = 0
EVAL_STEPS = 100  # Steps for policy evaluation
LAMBDA_THRESHOLD = 0.01

# 정규식으로 파라미터 추출
K_ITERATIONS = int(re.search(r"iter(\d+)", folder_name).group(1))
RND_BALANCE = re.search(r"balance_(True|False)", folder_name).group(1) == "True"
EPOCH = int(re.search(r"epoch_(\d+)", folder_name).group(1))
ALPHA  = float(re.search(r"alpha_([0-9.]+)", folder_name).group(1))
MIN_DEMO_TIME = int(re.search(r"minDemo(\d+)", folder_name).group(1))
HISTORIC_CONTEXT_LENGTH = int(re.search(r"H(\d+)", folder_name).group(1))
FREEZE = re.search(r"F(True|False)", folder_name).group(1) == "True"
ALG =re.search(r"\[(.*?)\]", folder_name).group(1)
SEED = int(re.search(r'seed(\d+)', folder_name).group(1))

print("===== Parsed Parameters from Folder Name =====")
print(f"K_ITERATIONS               : {K_ITERATIONS}")
print(f"RND_BALANCE                : {RND_BALANCE}")
print(f"EPOCH                      : {EPOCH}")
print(f"ALPHA                      : {ALPHA}")
print(f"MIN_DEMO_TIME              : {MIN_DEMO_TIME}")
print(f"HISTORIC_CONTEXT_LENGTH   : {HISTORIC_CONTEXT_LENGTH}")
print(f"FREEZE                     : {FREEZE}")
print("ALGORITHM                    :", ALG)
print("SEED                           :", SEED)
print("==============================================")

 # path where the model to be restored (resumed)
restored_folder_name = f"{folder_name}"
checkpoint_dir = os.path.join(os.path.dirname(folder_path), restored_folder_name)
os.makedirs(checkpoint_dir, exist_ok=True)


print("="*60)
print("RND-DAgger Training")
print("="*60)
print(f"Hyperparameters:")
print(f"  Lambda: {LAMBDA_THRESHOLD}")
print(f"  Min Demo Time: {MIN_DEMO_TIME}")
print(f"  Historic Context Length (H): {HISTORIC_CONTEXT_LENGTH}")
print(f"  Checkpoint directory: {checkpoint_dir}")
print("="*60)

# Get dimensions from environment
obs_dim = env.observation_space.shape[0]  # 25
action_dim = env.action_space.shape[0]    # 8
# ood_logger = OODLogger("/AILAB-summer-school-2025/RND/ood_log.csv")

print(f"\nObservation dim: {obs_dim}, Action dim: {action_dim}")

# Initialize expert policy
expert = ExpertPolicy(
    dt=set_env_dataCollection.env_cfg.sim.dt * set_env_dataCollection.env_cfg.decimation,
    num_envs=env.unwrapped.num_envs,
    device=device,
    env=env.unwrapped
)

# Initialize networks (RND input dimension depends on historic context)
policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
f_targ, f_pred = create_rnd_networks(
    obs_dim, 
    historic_context_length=HISTORIC_CONTEXT_LENGTH,
    output_dim=32, 
    device=device,
    freeze=FREEZE,
    seed= SEED,
)

policy.load_state_dict(
    torch.load(os.path.join(folder_path, f"policy_iter_{iter_num}.pt"), 
    map_location=device)
)

f_pred.load_state_dict(
    torch.load(os.path.join(folder_path, f"f_pred_iter_{iter_num}.pt"),
    map_location=device)
)

f_targ.load_state_dict(
    torch.load(os.path.join(folder_path, f"f_targ_iter_{iter_num}.pt"),
    map_location=device)
)

start_iter = iter_num + 1
# Initialize RND-DAgger
dagger = RNDDAgger(
    policy=policy,
    f_targ=f_targ,
    f_pred=f_pred,
    expert=expert,
    device=device,
    lambda_threshold=LAMBDA_THRESHOLD,
    min_demo_time=MIN_DEMO_TIME,
    historic_context_length=HISTORIC_CONTEXT_LENGTH,
    policy_lr=3e-4,
    rnd_lr=1e-4,
    batch_size=256,
    logger=None, # ood logger 
    alg = ALG
)
data = np.load(os.path.join(folder_path, f"expert_dataset_{iter_num}.npz"))

states = data["states"]
actions = data["actions"]
dagger.dataset.states.append(torch.tensor(states))
dagger.dataset.actions.append(torch.tensor(actions))
# dagger.dataset.expert_actions.append(torch.tensor(data["expert_actions"], dtype=torch.float32))
print("Dataset length:", len(dagger.dataset))
# Save hyperparameters
hyperparams = {
    'lambda_threshold': LAMBDA_THRESHOLD,
    'min_demo_time': MIN_DEMO_TIME,
    'historic_context_length': HISTORIC_CONTEXT_LENGTH,
    'k_iterations': K_ITERATIONS,
    'steps_per_iteration': STEPS_PER_ITERATION,
    'initial_expert_steps': INITIAL_EXPERT_STEPS,
    'obs_dim': obs_dim,
    'action_dim': action_dim,
}
with open(f'{checkpoint_dir}/hyperparameters.json', 'w') as f:
    json.dump(hyperparams, f, indent=4)

# ==========================
# Load previous results.json
# ==========================
prev_results_path = os.path.join(folder_path, "results.json")

if os.path.exists(prev_results_path):
    print(f"✅ Loading previous results from {prev_results_path}")
    with open(prev_results_path, "r") as f:
        results = json.load(f)
else:
    print("⚠️ No previous results.json found. Creating new one.")
    results = {
        'iterations': [],
        'dataset_sizes': [],
        'nswitches': [],
        'policy_losses': [],
        'rnd_losses': [],
        'eval_rewards': [],
        'eval_episodes': []
    }


print("\n" + "="*60)
print("STEP 1: Collecting initial expert dataset D")
print("="*60)

num_samples = dagger.collect_expert_only_data(env, INITIAL_EXPERT_STEPS)
print(f"Collected {num_samples} expert samples")
print(f"Initial dataset size: {len(dagger.dataset)}")

print("\n" + "="*60)
print("STEP 2: Training initial policy π₀ on expert dataset D")
print("="*60)

# print("Pre-training policy on expert dataset...")
# for epoch in range(50):
#     policy_loss = dagger.train_policy(num_epochs=1)
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}/50 - Policy loss: {policy_loss:.4f}")

# print(f"Initial policy training complete. Final loss: {policy_loss:.4f}")

torch.save(policy.state_dict(), f'{checkpoint_dir}/policy_pi0.pt')
print("Saved initial policy π₀")

print("\n" + "="*60)
print("STEP 3: Starting RND-DAgger iterations")
print("="*60)

# Training loop
for iteration in range(start_iter, K_ITERATIONS+1):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{K_ITERATIONS}")
    print(f"{'='*60}")
    
    beta = dagger.get_beta(iteration, K_ITERATIONS)
    beta = 0 # evaluate 

    # 1. Evaluate policy (no expert intervention)
    eval_reward, eval_episodes = dagger.evaluate_policy(env, num_steps=EVAL_STEPS)
    
    # 2. Collect data with expert intervention
    print("\nCollecting data with RND-based expert intervention...")
    num_samples = dagger.collect_data(env, STEPS_PER_ITERATION, beta, alpha_=ALPHA)
    
    dataset_size = len(dagger.dataset)
    nswitch = dagger.nswitch
    
    print(f"Collected {num_samples} samples")
    print(f"Total dataset size: {dataset_size}")
    print(f"Total context switches: {nswitch}")
    
    # 3. Train policy
    print("\nTraining policy...")
    policy_loss = dagger.train_policy(num_epochs=10)
    print(f"Policy loss: {policy_loss:.4f}")
    
    # 4. Train RND
    print("Training RND predictor...")
    if RND_BALANCE:
        rnd_loss = dagger.train_rnd_balanced(
        num_epochs=EPOCH,
        n_proj=8,
        eps=0.1,
        seed=SEED,
        )
    else:
        rnd_loss = dagger.train_rnd(num_epochs=5)
    print(f"RND loss: {rnd_loss:.4f}")
    
    # Store results
    results['iterations'].append(iteration + 1)
    results['dataset_sizes'].append(dataset_size)
    results['nswitches'].append(nswitch)
    results['policy_losses'].append(policy_loss)
    results['rnd_losses'].append(rnd_loss)
    results['eval_rewards'].append(eval_reward)
    results['eval_episodes'].append(eval_episodes)


    print(f"\nIteration {iteration + 1} Summary:")
    print(f"  Eval Reward: {eval_reward:.4f}")
    print(f"  Dataset Size: {dataset_size}")
    print(f"  Context Switches: {nswitch}")
    print(f"  Policy Loss: {policy_loss:.4f}")
    print(f"  RND Loss: {rnd_loss:.4f}")
    
    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        torch.save(policy.state_dict(), f'{checkpoint_dir}/policy_iter_{iteration + 1}.pt')
        print(f"\nSaved policy checkpoint at iteration {iteration + 1}")
    
    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        torch.save(f_pred.state_dict(), f'{checkpoint_dir}/f_pred_iter_{iteration + 1}.pt')
        print(f"\nSaved policy checkpoint at iteration {iteration + 1}")

    import numpy as np 
    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        torch.save(f_targ.state_dict(), f'{checkpoint_dir}/f_targ_iter_{iteration + 1}.pt')
        print(f"\nSaved f_target checkpoint at iteration {iteration + 1}")
        
    if (iteration + 1) % 5 == 0:
        import glob 
        pattern = os.path.join(glob.escape(checkpoint_dir), "expert_dataset_*.npz")
        old_files = glob.glob(pattern)
        for f in old_files:
            os.remove(f) # remove old files 
        np.savez_compressed(
            f"{checkpoint_dir}/expert_dataset_{iteration + 1}.npz",
            states=torch.cat(dagger.dataset.states).numpy(),
            actions=torch.cat(dagger.dataset.actions).numpy(),
            expert_actions=torch.cat(dagger.dataset.expert_actions).numpy()
        )
        print(f"\nSaved dataset at iteration {iteration + 1}")

    # Save results after each iteration
    with open(f'{checkpoint_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)


print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Final dataset size: {len(dagger.dataset)}")
print(f"Total context switches: {dagger.nswitch}")
print(f"Final eval reward: {results['eval_rewards'][-1]:.4f}")

# Save final model
torch.save(policy.state_dict(), f'{checkpoint_dir}/final_policy.pt')
torch.save({
    'policy_state_dict': policy.state_dict(),
    'f_pred_state_dict': f_pred.state_dict(),
    'results': results,
    'hyperparameters': hyperparams,
}, f'{checkpoint_dir}/final_checkpoint.pt')

print(f"\nAll results saved to: {checkpoint_dir}")
print("="*60)