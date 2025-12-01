import os
import re
import json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skrl.utils import set_seed
from env import set_env_dataCollection
from task_utils.setting_config import device, env
from models.bc_policy import BCPolicy

# =========================================================
# 1. evaluation function (user provided logic)
# =========================================================
def evaluate_policy(policy, env, num_episodes=50):
    env.unwrapped.reset()
    obs, _ = env.reset()
    success_count = 0
    episode_count = 0
    
    policy.eval()

    while episode_count < num_episodes:
        with torch.no_grad():
            action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        if done.any():
            episode_count += done.sum().item()

            log = info.get("log", {})
            success_flag = log.get("Episode_Termination/object_inside_bin", 0)
            if success_flag == 1:
                success_count += 1

            obs, _ = env.reset()
    print(f"Evaluation complete: Success Rate = {success_count / episode_count:.4f}, Episodes = {episode_count}")
    return success_count / episode_count

def extract_alpha(folder_name):
    match = re.search(r"alpha_(\d*\.?\d+)", folder_name)
    return float(match.group(1)) if match else None

def evaluate_all_iters(
    checkpoint_root,
    algorithms,
    seeds,
    num_episodes,
    save_json_path,
    env,
    device,
):

    results = {}

    # 폴더 패턴: [Balanced_Dagger]....seed42
    folder_pattern = r"\[(.*?)\].*seed(\d+)"

    folders = os.listdir(checkpoint_root)

    print("\n====================================")
    print(" FULL ITER EVALUATION START ")
    print("====================================\n")

    for folder in tqdm(folders):
        match = re.match(folder_pattern, folder)
        if not match:
            continue

        algo_name, seed = match.group(1), int(match.group(2))

        if algo_name not in algorithms or seed not in seeds:
            continue

        folder_path = os.path.join(checkpoint_root, folder)

        # alpha 추출
        alpha = extract_alpha(folder)

        # JSON 내 계층 만들기
        if algo_name not in results:
            results[algo_name] = {"alpha": alpha, "seeds": {}}

        if str(seed) not in results[algo_name]["seeds"]:
            results[algo_name]["seeds"][str(seed)] = {"iter_results": {}}

        # --------------------------------------------------------
        # iter list 얻기 (policy_iter_XX.pt)
        # --------------------------------------------------------
        policy_files = [
            f for f in os.listdir(folder_path)
            if f.startswith("policy_iter_") and f.endswith(".pt")
        ]

        #iter_list = sorted([int(f.split("_")[-1].split(".")[0]) for f in policy_files])
        iter_list = sorted([
        int(f.split("_")[-1].split(".")[0]) 
        for f in policy_files
        if int(f.split("_")[-1].split(".")[0]) <= 100
        ])

        # --------------------------------------------------------
        # iter별 evaluate 수행
        # --------------------------------------------------------
        for it in iter_list:
            ckpt_name = f"policy_iter_{it}.pt"
            ckpt_path = os.path.join(folder_path, ckpt_name)

            print(f"\n▶ Evaluating {algo_name} | seed={seed} | iter={it}")
            print(f"   checkpoint = {ckpt_path}")

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128,128,128]).to(device)
            policy.load_state_dict(torch.load(ckpt_path, map_location=device))

            success_rate = evaluate_policy(policy, env, num_episodes)

            # JSON에 저장
            results[algo_name]["seeds"][str(seed)]["iter_results"][str(it)] = float(success_rate)

    # --------------------------------------------------------
    # JSON 저장
    # --------------------------------------------------------
    with open(save_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n====================================")
    print(" Evaluation Done & Saved!")
    print(" File:", save_json_path)
    print("====================================")

    return results

def plot_results(save_json_path, target_iter):
    with open(save_json_path, "r") as f:
        data = json.load(f)

    algos = list(data.keys())
    means, stds = [], []

    for algo in algos:
        vals = list(data[algo].values())
        means.append(np.mean(vals) if len(vals) > 0 else 0.0)
        stds.append(np.std(vals) if len(vals) > 0 else 0.0)

    plt.figure(figsize=(8,6))
    x = np.arange(len(algos))

    plt.bar(x, means, yerr=stds, capsize=8)
    plt.xticks(x, algos, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Success Rate")

    if target_iter is None:
        plt.title("Policy Success Rate (Latest Iter, mean ± std)")
        output_name = "success_rate_latest.png"
    else:
        plt.title(f"Policy Success Rate (iter={target_iter}, mean ± std)")
        output_name = f"success_rate_iter{target_iter}.png"

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.show()

    print(f"Saved plot → {output_name}")

from task_utils.setting_config import env, device
from models.bc_policy import BCPolicy

CHECKPOINT_ROOT = "/AILAB-summer-school-2025/checkpoints/total"

results = evaluate_all_iters(
    checkpoint_root=CHECKPOINT_ROOT,
    algorithms=["Pure_Dagger", "R_Dagger", "Balanced_Dagger", "Safe_Dagger"],
    seeds=[2],
    num_episodes=5,
    save_json_path="eval_all_iters.json",
    env=env,
    device=device,
)