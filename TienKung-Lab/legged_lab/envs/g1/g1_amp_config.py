# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

"""
G1 AMP Environment Configuration

纯 AMP (Adversarial Motion Priors) 配置，不依赖 DWAQ VAE encoder。

Architecture:
- Actor: obs (含高度扫描) -> actions        (非盲行走，actor 可见地形)
- Critic: privileged_obs -> value
- Discriminator: AMP state pairs -> expert_score

AMP Observations (58 dimensions):
- joint_pos: 29 dims  (all DOF positions)
- joint_vel: 29 dims  (all DOF velocities)

Expert Data:
- Male2Walking_c3d: 23 x NPZ files, ~3734 frames of natural walking

Reward:
- total_reward = task_reward * (1 - amp_task_reward_lerp)
                + amp_reward * amp_task_reward_lerp
"""

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
import legged_lab.mdp as mdp
from isaaclab.envs.mdp import events as isaaclab_events

from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
from legged_lab.terrains import ROUGH_TERRAINS_CFG


# ============================================================
# Reward Configuration
# ============================================================

@configclass
class G1AmpRewardCfg(RewardCfg):
    """
    G1 AMP reward configuration.

    与 G1DwaqRewardCfg 一致，但 Actor 有高度扫描信息（非盲行走），
    因此对脚踝绊倒的惩罚可适当宽松。
    """

    # --- 速度追踪 ---
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5}
    )

    # --- 基础稳定性 ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # --- 能效 ---
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- 接触约束 ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names="(?!.*ankle.*).*"
            ),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 1.0,
        },
    )

    # --- 终止惩罚 ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # --- 步态质量 ---
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*ankle_roll.*"]
            ),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,  # AMP 有地形信息，放宽绊倒惩罚
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*ankle_roll.*"]
            )
        },
    )

    # --- 关节约束 ---
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"]
            )
        },
    )

    # --- 存活 & 偷懒惩罚 ---
    alive = RewTerm(func=mdp.alive, weight=0.15)
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )

    # --- 步态相位 ---
    gait_phase_contact = RewTerm(
        func=mdp.gait_phase_contact,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_ankle_roll.*", "right_ankle_roll.*"],
            ),
            "stance_threshold": 0.55,
        },
    )

    # --- 抬腿高度 ---
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=".*ankle_roll.*"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "target_height": 0.08,
        },
    )


# ============================================================
# Environment Configuration
# ============================================================

@configclass
class G1AmpEnvCfg(BaseEnvCfg):
    """
    G1 AMP environment configuration.

    与 G1DwaqEnvCfg 类似，但 Actor **可见**高度扫描（非盲行走）。
    AMP 判别器监督步态自然性。
    """

    reward = G1AmpRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [
            "left_ankle_roll.*",
            "right_ankle_roll.*",
        ]
        self.domain_rand.events.add_base_mass.params[
            "asset_cfg"
        ].body_names = [".*torso.*"]

        # Height scanner: Actor 也可以看（非 critic_only）
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.critic_only = False  # Actor 可见地形

        # 特权信息（Critic）
        self.scene.privileged_info.enable_feet_info = True
        self.scene.privileged_info.enable_feet_contact_force = True
        self.scene.privileged_info.enable_root_height = True

        # 不使用 DWAQ obs history（不需要 VAE encoder）
        self.robot.dwaq_obs_history_length = 1
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

        # 步态相位
        self.robot.gait_phase.enable = True
        self.robot.gait_phase.period = 0.8
        self.robot.gait_phase.offset = 0.5

        # 执行器增益随机化
        self.domain_rand.events.randomize_actuator_gains = EventTerm(
            func=isaaclab_events.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )


# ============================================================
# Agent Configuration
# ============================================================

@configclass
class G1AmpAgentCfg(BaseAgentCfg):
    """
    G1 AMP agent configuration.

    使用标准 ActorCritic + AMPPPO + G1AmpOnPolicyRunner。
    无 DWAQ VAE encoder，AMP 判别器提供步态先验。
    """

    experiment_name: str = "g1_amp"
    wandb_project: str = "g1_amp"
    runner_class_name: str = "G1AmpOnPolicyRunner"

    # AMP 顶层参数（由 G1AmpOnPolicyRunner 读取）
    amp_motion_files: list = None
    amp_num_preload_transitions: int = 300000
    amp_reward_coef: float = 0.02
    amp_task_reward_lerp: float = 0.1
    amp_discr_hidden_dims: list = None

    def __post_init__(self):
        super().__post_init__()

        # 标准 ActorCritic（非 DWAQ）
        self.policy.class_name = "ActorCritic"
        self.policy.init_noise_std = 1.0
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        # AMPPPO 算法
        self.algorithm.class_name = "AMPPPO"
        self.algorithm.entropy_coef = 0.01

        # ==================== AMP Expert Motion Files ====================
        npz_datasets_dir = (
            "/home/c211/WorkSpace/G1DWAQ_Lab/TienKung-Lab/"
            "legged_lab/envs/g1/datasets/Male2Walking_c3d"
        )
        if os.path.isdir(npz_datasets_dir):
            npz_files = sorted(
                [
                    os.path.join(npz_datasets_dir, f)
                    for f in os.listdir(npz_datasets_dir)
                    if f.endswith(".npz")
                ]
            )
            self.amp_motion_files = npz_files
            print(
                f"[G1AmpConfig] Found {len(npz_files)} npz files in {npz_datasets_dir}"
            )
        else:
            print(f"[WARNING] NPZ directory not found: {npz_datasets_dir}")
            self.amp_motion_files = []

        # ==================== AMP Hyperparameters ====================
        # amp_reward_coef: 判别器奖励权重
        self.amp_reward_coef = 0.02
        # amp_task_reward_lerp: AMP 奖励混合比例
        #   total_reward = task*(1-lerp) + amp*lerp
        #   复杂地形任务优先，AMP 占比不宜过高
        self.amp_task_reward_lerp = 0.1
        # 预加载帧数：134 文件约 2.5万帧，全量加载
        self.amp_num_preload_transitions = 300000
        # 判别器网络结构
        self.amp_discr_hidden_dims = [1024, 512, 256]
