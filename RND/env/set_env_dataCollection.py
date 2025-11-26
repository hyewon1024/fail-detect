import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed 


set_seed()  # e.g. `set_seed(42)` for fixed seed


# Isaac Lab 관련 라이브러리 임포트
from isaaclab.app import AppLauncher

# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

num_envs = 1 #please check headless. or it may cause a probnlem

# if num_envs > 20:
#     args_cli.headless = True

# else:
#     args_cli.headless = False
args_cli.headless =True # Env Off 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg


# 커스텀 환경 시뮬레이션 환경 config 파일 임포트
from env.PickAndPlace_env_cfg import PickAndPlaceEnvCfg
# gymnasium 라이브러리를 활용한 시뮬레이션 환경 선언
from env.config.franka.ik_abs_env_cfg import FrankaCubePickAndPlaceEnvCfg 

gym.register(
    id="Isaac-PickAndPlace-Cube-Franka-Custom-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubePickAndPlaceEnvCfg,
    },
    disable_env_checker=True,
)

# 환경 및 설정 파싱
env_cfg: PickAndPlaceEnvCfg = parse_env_cfg(
    "Isaac-PickAndPlace-Cube-Franka-Custom-v1",
    device=args_cli.device,
    num_envs=num_envs,
    use_fabric=not args_cli.disable_fabric,
   
)

def make_env(id = "Isaac-PickAndPlace-Cube-Franka-Custom-v1"):

    # 환경 생성 및 초기화
    env = gym.make(id, cfg=env_cfg)
    env = wrap_env(env)
    return env