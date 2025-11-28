# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def drop_to_bin(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    # object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin")
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]
    #  bin: RigidObject = env.scene[bin_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0:3, :]
    bin_pos = bin.data.root_pos_w
    object_pos = object.data.root_pos_w

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(bin_pos - object_pos, dim=1)
    # rewarded if the object is lifted above the threshold
    return (ee_w[0][0][0] > 0.0 and ee_w[0][0][0] < 0.5)*(ee_w[0][0][1] > 0.4 and ee_w[0][0][1] < 0.8) *(ee_w[0][0][2] > minimal_height) * (1 - torch.tanh(distance / std))

def object_in_goal(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    # 객체-빈 근접 판정 임계값
    threshold: float = 0.2,
    # EE-빈 근접 판정 임계값 (EE는 이 범위 "밖"이어야 보상)
    ee_threshold: float = 0.3,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    bin_obj: RigidObject = env.scene[bin_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    bin_pos = bin_obj.data.root_pos_w           # [N, 3]
    obj_pos = obj.data.root_pos_w               # [N, 3]
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]  # [N, 3]

    # 객체-빈 거리
    distance = torch.norm(bin_pos[:, :] - obj_pos[:, :], dim=1)   # [N]
    in_bin = (distance < threshold)        # [N] bool

    # EE-빈 거리
    d_ee_bin = torch.norm(ee_pos - bin_pos, dim=1)              # [N]
    ee_outside = d_ee_bin > ee_threshold                        # [N] bool

    # 조건: 객체는 bin 안, EE는 bin 밖
    reward_bool = in_bin & ee_outside                           # [N] bool
    return reward_bool.float()                                   # sparse: 0 or 1

def object_speed_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,   # 기준 속도
    small_reward: float = 0.1, # 임계값 이하 보상
    growth_rate: float = 2.0, # 지수 증가 속도
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    v = env.scene[object_cfg.name].data.root_lin_vel_w
    speed = torch.linalg.norm(v, dim=-1)

    # threshold 이하일 때: 작은 보상
    base = (speed <= threshold).float() * small_reward

    # threshold 초과일 때: 초과분에 지수 보상
    excess = torch.clamp(speed - threshold, min=0.0)
    exp_reward = (speed > threshold).float() * torch.exp(growth_rate * excess)

    return base + exp_reward

def obj_bin_dist(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    bin: RigidObject = env.scene[bin_cfg.name]

    object_pos = object.data.root_pos_w
    bin_pos = bin.data.root_pos_w

    dist = torch.norm(object_pos - bin_pos, dim=1)

    return 1-torch.tanh(dist/0.1)