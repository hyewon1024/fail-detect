# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def object_reached_goal(
    env: ManagerBasedRLEnv,
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    bin: RigidObject = env.scene[bin_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos = obj.data.root_pos_w[:, :3]   # (num_envs, 3)
    x = obj_pos[:, 0]
    y = obj_pos[:, 1]
    z = obj_pos[:, 2]

    # 지정한 직사각형 범위
    cond_x = (x >= 0.0) & (x <= 0.4)
    cond_y = (y >= 0.35) & (y <= 0.8)
    cond_z = (z >= 0.0) & (z <= 0.1)

    inside = cond_x & cond_y & cond_z     # 모든 조건을 만족해야 True

    return inside