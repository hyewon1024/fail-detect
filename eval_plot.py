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

def load_existing_results(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        return {}

def update_json(json_path, results):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[UPDATED] Saved JSON → {json_path}")

def evaluate_policy(policy, env, num_episodes=20):
    env.unwrapped.reset()
    obs, _ = env.reset()

    success_list = []
    reward_list = []

    policy.eval()

    while len(success_list) < num_episodes:
        episode_reward = 0.0

        while True:
            with torch.no_grad():
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            episode_reward+= reward.sum().item()

            if done.any():
                log = info.get("log", {})
                success_flag = log.get("Episode_Termination/object_inside_bin", 0)

                success_list.append(success_flag)
                reward_list.append(episode_reward)

                obs, _ = env.reset()
                break

    success_rate = np.mean(success_list)

    return success_rate, success_list, reward_list


def extract_alpha(folder_name):
    match = re.search(r"alpha_(\d*\.?\d+)", folder_name)
    return float(match.group(1)) if match else None

def evaluate_all_iters_incremental(
    checkpoint_root,
    algorithms,
    seeds,
    num_episodes,
    save_json_path,
    env,
    device,
):

    # =====================================================
    # 0. 기존 JSON 불러오기
    # =====================================================
    existing = load_existing_results(save_json_path)

    # 폴더 리스트
    folders = os.listdir(checkpoint_root)
    print(folders)
    
    # 폴더명 파싱 패턴
    # folder_pattern = r"\[(.*?)\].*seed(\d+)"

    print("\n==============================")
    print(" Incremental Evaluation Start ")
    print("==============================\n")

    for folder in tqdm(folders):
        match = re.search(r"\[(.*?)\].*?_seed(\d+)$", folder)

        if not match:
            continue

        algo_name, seed = match.group(1), int(match.group(2))
        print("PARSED FOLDER:", folder)
        print("  → algo:", algo_name, ", seed:", seed)

        if algo_name not in algorithms:
            continue

        if seed not in seeds:
            continue

        folder_path = os.path.join(checkpoint_root, folder)

        # α 추출
        alpha = extract_alpha(folder)

        # =====================================================
        #  alpha key 설정 (예: "0.5")
        # =====================================================
        alpha_key = str(alpha)

        if algo_name not in existing:
            existing[algo_name] = {}

        if alpha_key not in existing[algo_name]:
            existing[algo_name][alpha_key] = {"seeds": {}}

        if str(seed) not in existing[algo_name][alpha_key]["seeds"]:
            existing[algo_name][alpha_key]["seeds"][str(seed)] = {"iter_results": {}}
        recorded_iters = existing[algo_name][alpha_key]["seeds"][str(seed)]["iter_results"].keys()


        # ======================================================
        # policy_iter_xx.pt 추출
        # ======================================================
        policy_files = [
            f for f in os.listdir(folder_path)
            if f.startswith("policy_iter_") and f.endswith(".pt")
        ]

        iter_list = sorted([
            int(f.split("_")[-1].split(".")[0])
            for f in policy_files
            if int(f.split("_")[-1].split(".")[0]) <= 100
        ])

        # ======================================================
        #  ⬇ 기존 JSON에 없는 iter만 평가
        # ======================================================
        for it in iter_list:
            if str(it) in recorded_iters:
                continue

            ckpt_path = os.path.join(folder_path, f"policy_iter_{it}.pt")

            print(f"\n▶ NEW Evaluation Required")
            print(f"   Algo = {algo_name}, Seed = {seed}, Iter = {it}")
            print(f"   ckpt = {ckpt_path}")

            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
            policy.load_state_dict(torch.load(ckpt_path, map_location=device))

            success_rate, success_list, reward_list = evaluate_policy(policy, env, num_episodes)
            iter_data = {
                "success_rate": float(np.mean(success_list)),
                "success_mean": float(np.mean(success_list)),
                "success_std": float(np.std(success_list)),
                "success_list": success_list,
                "reward_mean": float(np.mean(reward_list)),
                "reward_std": float(np.std(reward_list)),
                "reward_list": reward_list,
            }
            existing[algo_name][alpha_key]["seeds"][str(seed)]["iter_results"][str(it)] = iter_data

            update_json(save_json_path, existing)

    print("\n==============================")
    print(" Incremental Eval Done! Saved")
    print("==============================")
    return existing

from task_utils.setting_config import env, device
from models.bc_policy import BCPolicy

CHECKPOINT_ROOT = "/AILAB-summer-school-2025/checkpoints/total"

results = evaluate_all_iters_incremental(
    checkpoint_root="/AILAB-summer-school-2025/checkpoints/total",
    algorithms=["Pure_Dagger", "R_Dagger", "Balanced_Dagger", "Safe_Dagger"],
    seeds=[2, 13, 42],
    num_episodes=15,
    save_json_path="/AILAB-summer-school-2025/eval_all_iters.json",
    env=env,
    device=device,
)