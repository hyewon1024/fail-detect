import os
import torch
import numpy as np

# ==========================================================
# 1) Matplotlib (WebAgg) for real-time plotting
# ==========================================================
import matplotlib
import matplotlib.pyplot as plt

# ==========================================================
# 2) Imports from your project
# ==========================================================
from env import set_env_dataCollection
from task_utils.setting_config import device, env
from task_utils.ExpertPolicy import ExpertPolicy

from models.bc_policy import BCPolicy
from models.rnd_network import create_rnd_networks
from algorithms.rnd_dagger_adaptive import RNDDAgger
class OODLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(log_path, "w") as f:
            f.write("global_step,lambda,ood,episode\n")

    def log(self, global_step, lambda_val, ood, episode):
        with open(self.log_path, "a") as f:
            f.write(f"{global_step},{lambda_val},{ood},{episode}\n")

    def log_episode_end(self, episode):
        with open(self.log_path, "a") as f:
            f.write(f"EPISODE_END,{episode}\n")

# ==========================================================
# 3) Helper: Load RNDDAgger from checkpoint directory
# ==========================================================
def load_rnd_dagger_agent(
    checkpoint_dir,
    obs_dim,
    action_dim,
    expert,
    device,
    lambda_threshold,
    min_demo_time,
    policy_lr=3e-3,
    rnd_lr=1e-3,
    batch_size=256
):

    print(f"\nLoading RNDDAgger agent from: {checkpoint_dir}")

    # Paths
    policy_ckpt = os.path.join(checkpoint_dir, "policy_iter_40.pt")
    f_pred_ckpt = os.path.join(checkpoint_dir, "f_pred_iter_40.pt")
    f_targ_ckpt = os.path.join(checkpoint_dir, "f_targ_iter_40.pt")

    # 1) Networks
    policy = BCPolicy(obs_dim, action_dim, hidden_dims=[128, 128, 128]).to(device)
    f_targ, f_pred = create_rnd_networks(obs_dim, historic_context_length=0, output_dim=32, device=device)

    # 2) Load weights
    print("  - Loading policy")
    policy.load_state_dict(torch.load(policy_ckpt, map_location=device))

    print("  - Loading f_pred")
    f_pred.load_state_dict(torch.load(f_pred_ckpt, map_location=device))

    print("  - Loading f_targ")
    f_targ.load_state_dict(torch.load(f_targ_ckpt, map_location=device))

    policy.eval()
    f_pred.eval()
    f_targ.eval()

    print("✓ Loaded all RNDDAgger components\n")
    ood_logger = OODLogger("/AILAB-summer-school-2025/RND/ood_log.csv")
    # 3) Create dagger agent
    dagger = RNDDAgger(
        policy=policy,
        f_targ=f_targ,
        f_pred=f_pred,
        expert=expert,
        device=device,
        lambda_threshold=lambda_threshold,
        min_demo_time=min_demo_time,
        policy_lr=policy_lr,
        rnd_lr=rnd_lr,
        batch_size=batch_size,
        logger=ood_logger,
    )

    return dagger, policy, f_pred, f_targ


# ==========================================================
# 4) Setup Real-time Plot
# ==========================================================
def setup_realtime_plot(lambda_threshold):

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    x_vals, y_vals = [], []
    (line,) = ax.plot([], [], 'b-', linewidth=2, label="m(ood)")

    ax.axhline(lambda_threshold, linestyle='--', color='red', label=f"λ={lambda_threshold}")

    ax.set_title("Real-time RND OOD Measure")
    ax.set_xlabel("Step")
    ax.set_ylabel("m(ood)")
    ax.grid(True)
    ax.legend()

    plt.show()

    return fig, ax, line, x_vals, y_vals


# ==========================================================
# 5) Main
# ==========================================================
if __name__ == "__main__":

    # ---------------------------------
    # Env / dims
    # ---------------------------------
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    expert = ExpertPolicy(
        dt=set_env_dataCollection.env_cfg.sim.dt * set_env_dataCollection.env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=device,
        env=env.unwrapped
    )

    checkpoint_dir = "/AILAB-summer-school-2025/RND/checkpoints/rnd_iter100_balance_True_epoch_10_alpha_0.95_minDemo70_H0_FTrue"
    # import re
    # match = re.search(r"lambda([0-9]*\.?[0-9]+)", checkpoint_dir)
    # if match:
    #     lambda_threshold= float(match.group(1))
    # else:
    #     raise ValueError("lambda value not found in checkpoint_dir")
    lambda_threshold= 0 

    dagger, policy, f_pred, f_targ = load_rnd_dagger_agent(
        checkpoint_dir=checkpoint_dir,
        obs_dim=obs_dim,
        action_dim=action_dim,
        expert=expert,
        device=device,
        lambda_threshold=lambda_threshold,
        min_demo_time=0
    )
    lambda_threshold = dagger.lambda_threshold
    # ---------------------------------
    # Logging setup
    # ---------------------------------
    log_path = "/AILAB-summer-school-2025/RND/ood_log.csv"
    with open(log_path, "w") as f:
        f.write("step, lambda, ood, episode\n")

    obs, _ = env.reset()
    step = 0
    episode_count = 1
    global_step = 0
    try:
        while True:

            with torch.no_grad():
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # --- Compute OOD ---
            m_vals = dagger.compute_ood_measure(obs)
            m_scalar = m_vals.mean().item()

            global_step += 1
            with open(log_path, "a") as f:
                f.write(f"{global_step},{dagger.lambda_threshold},{m_scalar},{episode_count}\n")

            if (terminated | truncated).any():
                print(f"[Episode {episode_count} END] Step={step}, OOD={m_scalar:.4f}")

                with open(log_path, "a") as f:
                    f.write(f"EPISODE_END,{episode_count}\n")

                episode_count += 1 
                step = 0                

                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    env.close()