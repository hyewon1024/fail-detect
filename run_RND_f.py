import torch
from skrl.utils import set_seed
from env import set_env_dataCollection
from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy
from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger_adaptive import RNDDAgger
import json
from collections.abc import Sequence
# from task_utils.ood_logger import OODLogger
import numpy as np
import os
import re
import argparse, shlex
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--kit_args", type=str, default="")
args = parser.parse_args()

kit_parser = argparse.ArgumentParser()
kit_parser.add_argument("--folder", type=str, default=None)
kit_parser.add_argument("--Iter_NUM", type=int, default=None)

kit_argv = shlex.split(args.kit_args)
kit_args = kit_parser.parse_args(kit_argv)


FOLDER = kit_args.folder
Iter_NUM = kit_args.Iter_NUM

# -----------------------------------------------------
# 2. folder 또는 Iter_NUM 둘 중 하나라도 None이면 종료
# -----------------------------------------------------
try:
    if FOLDER is None or Iter_NUM is None:
        raise ValueError("Either FOLDER or Iter_NUM is None.")
except Exception as e:
    print(f"[ERROR] Invalid KIT_ARGS: {e}")
    sys.exit(1)

print(f"[INFO] Resume folder: {FOLDER}")
print(f"[INFO] Resume iteration: {Iter_NUM}")

# -----------------------------------------------------
# 3. hyperparameters.json 로드
# -----------------------------------------------------
hyper_path = os.path.join(FOLDER, "hyperparameters.json")

try:
    with open(hyper_path, "r") as f:
        hp = json.load(f)
except Exception as e:
    print(f"[ERROR] Failed to load hyperparameters.json at {hyper_path}")
    print(e)
    sys.exit(1)

print("[INFO] Loaded hyperparameters:")
for k, v in hp.items():
    print(f"  {k}: {v}")

# -----------------------------------------------------
# 4. JSON에서 기존 변수 복구
# -----------------------------------------------------

# JSON 필드가 없으면 None이 아니라 명확한 에러를 내도록 처리
def get_hp(name, required=True, default=None):
    if name in hp:
        return hp[name]
    if required:
        print(f"[ERROR] Missing hyperparameter: {name}")
        sys.exit(1)
    return default

K_ITERATIONS = get_hp("k_iterations")
STEPS_PER_ITERATION = get_hp("steps_per_iteration")
INITIAL_EXPERT_STEPS = get_hp("initial_expert_steps")
EVAL_ITERS = get_hp("eval_iters")

LAMBDA_THRESHOLD = get_hp("lambda_threshold")
MIN_DEMO_TIME = get_hp("min_demo_time")
HISTORIC_CONTEXT_LENGTH = get_hp("historic_context_length")
FREEZE = bool(get_hp("freeze", required=False, default=False))  # optional
EPOCH = get_hp("epoch")
SEED = get_hp("seed")
ALPHA = get_hp("alpha")
ALG = get_hp("ALG")
BETA = get_hp("beta", required=False, default=None)
TAU = get_hp("tau", required=False, default=None)

# 추가적으로 obs_dim, action_dim도 복구
OBS_DIM = get_hp("obs_dim")
ACTION_DIM = get_hp("action_dim")

if ALG == "Pure_Dagger":
    RND_BALANCE = False
    LAMBDA_THRESHOLD = 0
    if BETA == None:
        BETA = 0.95

elif ALG == "R_Dagger":
    RND_BALANCE = False
    LAMBDA_THRESHOLD = 0.01  # change adaptively 

elif ALG == "Balanced_Dagger":
    RND_BALANCE = True
    LAMBDA_THRESHOLD = 0.01  # change adaptively 

elif ALG =="Safe_Dagger":
    RND_BALANCE = False
    LAMBDA_THRESHOLD = 0
# -----------------------------------------------------
# 5. 디버그 출력
# -----------------------------------------------------
print("\n========== FINAL RESUME CONFIG ==========")
print(f"FOLDER                  : {FOLDER}")
print(f"Iter_NUM                : {Iter_NUM}")
print(f"K_ITERATIONS            : {K_ITERATIONS}")
print(f"STEPS_PER_ITERATION     : {STEPS_PER_ITERATION}")
print(f"INITIAL_EXPERT_STEPS    : {INITIAL_EXPERT_STEPS}")
print(f"EVAL_ITERS              : {EVAL_ITERS}")
print(f"LAMBDA_THRESHOLD        : {LAMBDA_THRESHOLD}")
print(f"MIN_DEMO_TIME           : {MIN_DEMO_TIME}")
print(f"HISTORIC_CONTEXT_LENGTH : {HISTORIC_CONTEXT_LENGTH}")
print(f"FREEZE                  : {FREEZE}")
print(f"EPOCH                   : {EPOCH}")
print(f"SEED                    : {SEED}")
print(f"ALPHA                   : {ALPHA}")
print(f"ALG                     : {ALG}")
print(f"BETA                    : {BETA}")
print(f"TAU                     : {TAU}")
print(f"OBS_DIM                 : {OBS_DIM}")
print(f"ACTION_DIM              : {ACTION_DIM}")
print("==========================================\n")
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
    torch.load(os.path.join(FOLDER, f"policy_iter_{Iter_NUM}.pt"), 
    map_location=device)
)

f_pred.load_state_dict(
    torch.load(os.path.join(FOLDER, f"f_pred_iter_{Iter_NUM}.pt"),
    map_location=device)
)

f_targ.load_state_dict(
    torch.load(os.path.join(FOLDER, f"f_targ_iter_{Iter_NUM}.pt"),
    map_location=device)
)

start_iter = Iter_NUM #we add 1 after stage
# Initialize RND-DAgger
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
    beta_start=BETA, 
    alg = ALG,
)
data = np.load(os.path.join(FOLDER, f"expert_dataset_{Iter_NUM}.npz"))

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
    'eval_iters' : EVAL_ITERS,
    'epoch' : EPOCH,
    'seed' : SEED,
    'alpha' : ALPHA,
    'ALG' : ALG,
    'beta' : BETA,
    'tau' : TAU,
    'obs_dim': obs_dim,
    'action_dim': action_dim,
}
with open(f'{FOLDER}/hyperparameters.json', 'w') as f:
    json.dump(hyperparams, f, indent=4)

# ==========================
# Load previous results.json
# ==========================
prev_results_path = os.path.join(FOLDER, "results.json")

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


# print("\n" + "="*60)
# print("STEP 1: Collecting initial expert dataset D")
# print("="*60)

# num_samples = dagger.collect_expert_only_data(env, INITIAL_EXPERT_STEPS)
# print(f"Collected {num_samples} expert samples")
# we dont need to collect dataset
# print(f"Initial dataset size: {len(dagger.dataset)}")

# print("\n" + "="*60)
# print("STEP 2: Training initial policy π₀ on expert dataset D")
# print("="*60)


# print("Pre-training policy on expert dataset...")
# for epoch in range(50):
#     policy_loss = dagger.train_policy(num_epochs=1)
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}/50 - Policy loss: {policy_loss:.4f}")

# print(f"Initial policy training complete. Final loss: {policy_loss:.4f}")
# we dont need to train initial policy dataset

print("\n" + "="*60)
print("STEP 3: Starting RND-DAgger iterations")
print("="*60)

# Training loop
for iteration in range(start_iter, K_ITERATIONS+1):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{K_ITERATIONS}")
    print(f"{'='*60}")
    
    if BETA is not None:
        beta = dagger.get_beta(iteration, K_ITERATIONS)
    else:
        beta = None 

    # 1. Evaluate policy (no expert intervention)
    eval_reward, eval_episodes = dagger.evaluate_policy(env, num_episodes_target=EVAL_ITERS)
    
    # 2. Collect data with expert intervention
    print("\nCollecting data with RND-based expert intervention...")
    num_samples = dagger.collect_data(env, STEPS_PER_ITERATION, beta, alpha_=ALPHA, tau=TAU)
    
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
        torch.save(policy.state_dict(), f'{FOLDER}/policy_iter_{iteration + 1}.pt')
        print(f"\nSaved policy checkpoint at iteration {iteration + 1}")
    
    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        torch.save(f_pred.state_dict(), f'{FOLDER}/f_pred_iter_{iteration + 1}.pt')
        print(f"\nSaved policy checkpoint at iteration {iteration + 1}")

    import numpy as np 
    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        torch.save(f_targ.state_dict(), f'{FOLDER}/f_targ_iter_{iteration + 1}.pt')
        print(f"\nSaved f_target checkpoint at iteration {iteration + 1}")
        
    if (iteration + 1) % 5 == 0:
        import glob 
        pattern = os.path.join(glob.escape(FOLDER), "expert_dataset_*.npz")
        old_files = glob.glob(pattern)
        for f in old_files:
            os.remove(f) # remove old files 

        states = torch.cat(dagger.dataset.states, dim=0)
        actions = torch.cat(dagger.dataset.actions, dim=0)
        expert_actions = torch.cat(dagger.dataset.expert_actions, dim=0)

        save_path = f"{FOLDER}/expert_dataset_{iteration + 1}.npz"

        np.savez_compressed(
            save_path,
            states=states.cpu().numpy(),
            actions=actions.cpu().numpy(),
            expert_actions=expert_actions.cpu().numpy()
        )
        print(f"\nSaved dataset at iteration {iteration + 1}")

    # Save results after each iteration
    with open(f'{FOLDER}/results.json', 'w') as f:
        json.dump(results, f, indent=4)


print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Final dataset size: {len(dagger.dataset)}")
print(f"Total context switches: {dagger.nswitch}")
print(f"Final eval reward: {results['eval_rewards'][-1]:.4f}")

# Save final model
torch.save(policy.state_dict(), f'{FOLDER}/final_policy.pt')
torch.save({
    'policy_state_dict': policy.state_dict(),
    'f_pred_state_dict': f_pred.state_dict(),
    'results': results,
    'hyperparameters': hyperparams,
}, f'{FOLDER}/final_checkpoint.pt')

print(f"\nAll results saved to: {FOLDER}")
print("="*60)