import os
import re
import numpy as np
import torch

from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy
from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger import RNDDAgger
import matplotlib.pyplot as plt
from env import set_env_dataCollection

def compute_a_devi(folder_path, lambda_val):

    print(f"\n[INFO] Running for λ = {lambda_val}")

    # ------------------------
    # Load checkpoint file paths
    # ------------------------
    dataset_path = os.path.join(folder_path, "expert_dataset_40.npz")
    policy_path  = os.path.join(folder_path, "policy_iter_5.pt")
    f_pred_path  = os.path.join(folder_path, "f_pred_iter_5.pt")
    f_targ_path  = os.path.join(folder_path, "f_targ_iter_5.pt")

    for p in [dataset_path, policy_path, f_pred_path, f_targ_path]:
        assert os.path.exists(p), f"Missing: {p}"

    # ------------------------
    # Load dataset
    # ------------------------
    data = np.load(dataset_path)

    obs = torch.tensor(data["states"], dtype=torch.float32).to(device)
    expert_action = torch.tensor(data["expert_actions"], dtype=torch.float32).to(device)

    # ------------------------
    # Load models
    # ------------------------
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
    f_targ, f_pred = create_rnd_networks(obs_dim, output_dim=32, device=device)

    policy.load_state_dict(torch.load(policy_path, map_location=device))
    f_pred.load_state_dict(torch.load(f_pred_path, map_location=device))
    f_targ.load_state_dict(torch.load(f_targ_path, map_location=device))

    policy.eval()
    f_pred.eval()
    f_targ.eval()

    # ------------------------
    # Expert policy
    # ------------------------
    expert = ExpertPolicy(
        dt=set_env_dataCollection.env_cfg.sim.dt * set_env_dataCollection.env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=device,
        env=env.unwrapped
    )

    dagger = RNDDAgger(
        policy=policy,
        f_targ=f_targ,
        f_pred=f_pred,
        expert=expert,
        device=device,
        lambda_threshold=lambda_val,   # ✅ 중요
        min_demo_time=70,
        historic_context_length=0,
        policy_lr=3e-4,
        rnd_lr=1e-4,
        batch_size=256
    )

    # ------------------------
    # Compute OOD score
    # ------------------------
    with torch.no_grad():
        m = dagger.compute_ood_measure(obs).cpu().numpy()

    mask_ood = m > lambda_val
    mask_id  = ~mask_ood

    # ------------------------
    # Compute policy action
    # ------------------------
    with torch.no_grad():
        act = dagger.policy(obs)

    # ------------------------
    # Action deviation
    # ------------------------
    diff = torch.norm(expert_action - act, dim=1).cpu().numpy()

    return {
        "lambda": lambda_val,
        "diff_id": diff[mask_id],
        "diff_ood": diff[mask_ood],
        "num_id": mask_id.sum(),
        "num_ood": mask_ood.sum()
    }

# def plot(folder_paths, save_path="/AILAB-summer-school-2025/RND/results"):
    
#     colors = plt.cm.viridis(np.linspace(0,1,len(folder_paths)))

#     plt.figure(figsize=(8,4))

#     # -----------------------------------------
#     # 1) ID Plot
#     # -----------------------------------------
#     plt.subplot(1,2,1)
#     for idx, folder in enumerate(folder_paths):
#         result = compute_a_devi(folder)
#         print(result)
#         diff_id = result["diff_id"]
#         lam = result["lambda"]

#         if len(diff_id) == 0:
#             continue 

#         p99 = np.percentile(diff_id, 99) if len(diff_id) > 0 else 0
#         bins = np.linspace(0, p99, 40)
#         hist, edges = np.histogram(diff_id, bins=bins)
#         centers = (edges[:-1] + edges[1:]) / 2.

#         total = len(result["diff_id"]) + len(result["diff_ood"])
#         count_id = len(result["diff_id"])
#         pct = (count_id / total * 100)

#         plt.plot(
#             centers, hist,
#             color=colors[idx],
#             label=f"λ={lam} ({count_id}/{total}, {pct:.1f}%)"
#         )


#     plt.title("ID Data - Action Deviation", fontsize=10)
#     plt.xlabel("‖expert − action‖₂")
#     plt.ylabel("count")
#     plt.legend(fontsize=6)
#     plt.grid(True)


#     # -----------------------------------------
#     # 2) OOD Plot
#     # -----------------------------------------
#     plt.subplot(1,2,2)
#     for idx, folder in enumerate(folder_paths):
#         result = compute_a_devi(folder)
#         diff_ood = result["diff_ood"]
#         lam = result["lambda"]

#         if len(diff_ood) == 0:
#             continue

#         p99 = np.percentile(diff_ood, 99) if len(diff_ood) > 0 else 0
#         bins = np.linspace(0, p99, 40)
#         hist, edges = np.histogram(diff_ood, bins=bins)
#         centers = (edges[:-1] + edges[1:]) / 2.

#         total = len(result["diff_id"]) + len(result["diff_ood"])
#         count_ood = len(result["diff_ood"])
#         pct = (count_ood / total * 100)

#         plt.plot(
#             centers, hist,
#             color=colors[idx],
#             label=f"λ={lam} ({count_ood}/{total}, {pct:.1f}%)"
#         )

#     plt.title("OOD Data - Action Deviation", fontsize=10)
#     plt.xlabel("‖expert − action‖₂")
#     plt.ylabel("count")
#     plt.legend(fontsize=6)
#     plt.grid(True)

#     plt.tight_layout()
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()  
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")

#     print(f"[✓] Saved PNG to: {save_path}")
def plot(folder_path, save_path):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ✅ 직접 지정할 lambda 리스트
    LAMBDA_LIST = [0.2, 0.1, 0.09, 0.08, 0.07]

    # ✅ 색상 고정
    lambda_colors = {
        0.2:  "red",
        0.1:  "orange",
        0.09: "green",
        0.08: "blue",
        0.07: "purple"
    }

    # =====================================================
    # 1) 각 lambda에 대해 데이터 계산
    # =====================================================
    all_results = []
    all_values = []

    for lam in LAMBDA_LIST:
        result = compute_a_devi(folder_path, lam)
        all_results.append(result)

        if len(result["diff_id"]) > 0:
            all_values.extend(result["diff_id"])
        if len(result["diff_ood"]) > 0:
            all_values.extend(result["diff_ood"])

    if len(all_values) == 0:
        print("[WARNING] No valid data found.")
        return

    # ============================================
    # 2) Bin 범위 설정
    # ============================================
    fixed_min = 0
    fixed_max = np.percentile(all_values, 99)  # 자동 맞춤
    bins = np.linspace(fixed_min, fixed_max, 40)

    plt.figure(figsize=(8, 4))

    # ============================================
    # 3) ID Plot
    # ============================================
    plt.subplot(1, 2, 1)

    for result in all_results:

        diff_id = result["diff_id"]
        lam = result["lambda"]

        if len(diff_id) == 0:
            continue

        color = lambda_colors[lam]

        id_mean = np.mean(diff_id)
        id_std  = np.std(diff_id)

        hist, edges = np.histogram(diff_id, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        total = result["num_id"] + result["num_ood"]
        count_id = result["num_id"]
        pct = (count_id / total * 100) if total > 0 else 0

        plt.plot(
            centers, hist,
            color=color,
            linewidth=0.8,
            alpha=1.0,
            label=f"λ={lam} | μ={id_mean:.3f}, σ={id_std:.3f} ({count_id}/{total}, {pct:.1f}%)"
        )

    plt.title("ID Data - Action Deviation", fontsize=11)
    plt.xlabel("‖expert − action‖₂")
    plt.ylabel("count")
    plt.legend(fontsize=7)
    plt.grid(True)

    # ============================================
    # 4) OOD Plot
    # ============================================
    plt.subplot(1, 2, 2)

    for result in all_results:

        diff_ood = result["diff_ood"]
        lam = result["lambda"]

        if len(diff_ood) == 0:
            continue

        color = lambda_colors[lam]

        ood_mean = np.mean(diff_ood)
        ood_std  = np.std(diff_ood)

        hist, edges = np.histogram(diff_ood, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        total = result["num_id"] + result["num_ood"]
        count_ood = result["num_ood"]
        pct = (count_ood / total * 100) if total > 0 else 0

        plt.plot(
            centers, hist,
            color=color,
            linewidth=0.8,
            alpha=1.0,
            label=f"λ={lam} | μ={ood_mean:.3f}, σ={ood_std:.3f} ({count_ood}/{total}, {pct:.1f}%)"
        )

    plt.title("OOD Data - Action Deviation", fontsize=11)
    plt.xlabel("‖expert − action‖₂")
    plt.ylabel("count")
    plt.legend(fontsize=7)
    plt.grid(True)

    plt.tight_layout()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Saved PNG to: {save_path}")

folder_folder_list = [
    # "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_lambda0_minDemo70_H0_FTrue",
    # "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter70_lambda0.1_V1",
    # "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_lambda0.01_minDemo70_H0_FTrue",
    # "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_lambda0.25_minDemo70_H0_FTrue/",
    "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_balance_True_epoch_10_alpha_0.95_minDemo70_H0_FTrue/",
]
plot("/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_balance_True_epoch_10_alpha_0.95_minDemo70_H0_FTrue/", "/AILAB-summer-school-2025/RND/results_50")