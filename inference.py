import torch
from skrl.utils import set_seed
from env import set_env_dataCollection
from task_utils.setting_config import device, env
from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger import RNDDAgger
import os


# ------------------------------------------------------
# 1. 기본 설정
# ------------------------------------------------------
set_seed(42)

# Load environment
print("Initializing environment...")
num_envs = env.unwrapped.num_envs
obs_dim = env.observation_space.shape[0]  
action_dim = env.action_space.shape[0] 
print(f"Observation dim = {obs_dim}, Action dim = {action_dim}")

# ------------------------------------------------------
# 2. 모델 초기화 및 체크포인트 로드
# ------------------------------------------------------
checkpoint_path = "/AILAB-summer-school-2025/RND/checkpoints/rnd_lambda0.001_minDemo50_H5/policy_iter_60.pt"
assert os.path.exists(checkpoint_path), f"❌ Checkpoint not found: {checkpoint_path}"

# 네트워크 생성
policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
f_targ, f_pred = create_rnd_networks(obs_dim, output_dim=128, device=device)

# checkpoint 로드
ckpt = torch.load(checkpoint_path, map_location=device)

policy.load_state_dict(torch.load("/AILAB-summer-school-2025/RND/checkpoints/rnd_lambda0.001_minDemo50_H5/policy_iter_60.pt", map_location=device)["policy_state_dict"])
f_pred.load_state_dict(torch.load("/AILAB-summer-school-2025/RND/checkpoints/rnd_lambda0.001_minDemo50_H5/policy_iter_60.pt", map_location=device)["f_pred_state_dict"])


print(f"✅ Loaded checkpoint from {checkpoint_path}")

# ------------------------------------------------------
# 3. RND-DAgger 객체 생성 (Expert 없음)
# ------------------------------------------------------
dagger = RNDDAgger(
    policy=policy,
    f_targ=f_targ,
    f_pred=f_pred,
    expert=None,                 
    device=device,
    lambda_threshold=1.0,
    min_demo_time=10,
    policy_lr=3e-3,
    rnd_lr=1e-3,
    batch_size=32,
)

# ------------------------------------------------------
# 4. 평가 루프
# ------------------------------------------------------
print("🚀 Starting policy evaluation (no expert)...")

avg_reward, num_episodes = dagger.evaluate_policy(env, num_steps=1000)

print("===========================================")
print(f"✅ Evaluation finished")
print(f"Average Reward: {avg_reward:.4f}")
print(f"Completed Episodes: {num_episodes}")
print("===========================================")
