# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
#
import glob
import random
import os
from isaaclab.sim.spawners import materials


@configclass
class CustomUsdFileCfg(UsdFileCfg):
    """커스텀 USD 파일 config - 물리 소재 경로를 지정하기 위함(기존의 UsdFileCfg 에는 물리 소재가 없음)"""

    # Prim에 적용할 물리 소재 경로 (상대 경로 가능)
    physics_material_path: str = "material"

    # 물리 소재를 명시적으로 지정. None이면 적용 안 함.
    physics_material: materials.PhysicsMaterialCfg | None = None

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the PickAndPlace scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
                size=(1.6, 2.0, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.2, roughness=0.5),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.5, restitution=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        # init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.4, 0.25)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.4, -0.25)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        #init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.55]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 바구니(Bin) 오브젝트
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        #init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.6, 0.555), rot=[0.7071, 0.7071, 0, 0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.6, 0.055), rot=[0.7071, 0.7071, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            #usd_path='../data/assets/basket/basket.usd',
            #for debug
            usd_path='data/assets/basket/basket.usd',
            scale=(0.8, 0.25, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.7, 0.5), metallic=0.2, roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        ),
    )



@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_desired_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.25), pos_y=(0.55, 0.60), pos_z=(0.3, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        bin_position = ObsTerm(func=mdp.bin_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_desired_pose"})
        gripper_position = ObsTerm(func=mdp.gripper_position)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.6, 0.6), "y": (-0.1, 0.1), "z": (0.0, 0.0)}, 
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Dense shaping: reach, lift, transport, and place the object in the bin area.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=4.0)

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=6.0,
    )

    transport_to_goal = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.25, "minimal_height": 0.04, "command_name": "object_desired_pose"},
        weight=10.0,
    )

    obj_dist = RewTerm(
        func=mdp.obj_bin_dist,
        weight=15.0,
    )

    place_success = RewTerm(func=mdp.object_in_goal, weight=50.0)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    object_inside_bin = DoneTerm(
        func=mdp.object_reached_goal
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    #1st
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 30000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 30000}
    # )

    # object_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_vel", "weight": -1e-6, "num_steps": 30000}
    # )
    # #2nd
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1, "num_steps": 60000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1, "num_steps": 60000}
    # )

    # object_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_vel", "weight": -1e-4, "num_steps": 60000}
    # )


##
# Environment configuration
##


@configclass
class PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the PickAndPlace environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5) #4096
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
