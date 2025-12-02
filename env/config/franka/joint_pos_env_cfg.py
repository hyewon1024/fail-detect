# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
# from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from env.PickAndPlace_env_cfg import PickAndPlaceEnvCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sim import PinholeCameraCfg               # 핀홀 카메라 모델 설정


@configclass
class FrankaCubePickAndPlaceEnvCfg(PickAndPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.reset_object_position = None

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_desired_pose.body_name = "panda_hand"
        import random
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=([4.0280e-01, -3.6916e-02, 0.055] + np.random.normal(0, 0.015, 3)).tolist(), rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # # 4. 카메라 센서 설정 (handeye 카메라: 프랑카 손 끝에 부착)
        # self.scene.camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand/handeye_camera",      # 카메라가 위치할 prim 경로
        #     update_period=0.2,      # 시뮬레이션 업데이트 간격(초)
        #     height=84, width=84,  # 해상도
        #     data_types=["rgb", "distance_to_image_plane"],      # RGB+Depth
        #     spawn=PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        #     # 카메라 위치/방향 오프셋 (ROS convention, Z축 90도 회전)
        #     offset=CameraCfg.OffsetCfg(pos=(0.1, 0.035, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="ros"),
        # )

        # self.scene.front_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/front_camera",     # 카메라가 위치할 prim 경로
        #     update_period=0.2,      # 시뮬레이션 업데이트 간격(초)
        #     height=84, width=84,  # 해상도
        #     data_types=["rgb"],      # RGB+Depth
        #     spawn=PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        # )



        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubePickAndPlaceEnvCfg_PLAY(FrankaCubePickAndPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
