# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

"""
G1 DWAQ + AMP Environment Configuration

This configuration combines:
1. DWAQ (Deep Variational Autoencoder for Walking): Self-supervised terrain adaptation via β-VAE
2. AMP (Adversarial Motion Priors): Supervised natural gait learning via discriminator

Key Features:
- DWAQ: Learns terrain-aware latent representation from observation history
- AMP: Matches expert motion data (pace1.txt) for natural walking
- Hybrid reward: task_reward * (1-lerp) + amp_reward * lerp

Architecture:
- Actor: obs + latent_code -> actions
- Encoder (VAE): obs_history -> latent_code + velocity_estimate
- Discriminator: state_pairs -> expert_score
- Critic: privileged_obs -> value

Expert Data:
- pace1.txt: 10.33s walking motion (492 frames @ 0.021s/frame)
- Format: base_pos(3) + base_quat(4) + dof_pos(29) + dof_vel(25) = 61 dims
"""

from legged_lab.envs.g1.g1_dwaq_config import G1DwaqEnvCfg, G1DwaqAgentCfg
from isaaclab.utils import configclass


@configclass
class G1DwaqAmpEnvCfg(G1DwaqEnvCfg):
    """
    G1 DWAQ + AMP environment configuration.

    Inherits all DWAQ environment settings:
    - Blind walking (height_scanner critic_only=True)
    - Observation history for VAE encoder (dwaq_obs_history_length=5)
    - Gait phase tracking for bipedal coordination
    - Privileged information for critic

    AMP adds expert motion matching on top of DWAQ terrain adaptation.
    """
    pass  # All environment settings inherited from G1DwaqEnvCfg


@configclass
class G1DwaqAmpAgentCfg(G1DwaqAgentCfg):
    """
    G1 DWAQ + AMP agent configuration.

    Combines DWAQ and AMP training:
    1. DWAQ Components (from parent):
       - ActorCritic_DWAQ policy with VAE encoder
       - cenet_out_dim = 19 (velocity 3 + latent 16)
       - Autoencoder loss (velocity + reconstruction + KL divergence)

    2. AMP Components (added):
       - Discriminator network for expert motion matching
       - AMP reward coefficient and task reward lerp
       - Motion replay buffer with preloaded transitions

    Training Loss:
    total_loss = ppo_loss + autoencoder_loss + amp_loss

    Hybrid Reward:
    total_reward = task_reward * (1-lerp) + amp_reward * lerp
    - lerp=0.3: 70% task reward (terrain) + 30% AMP reward (natural gait)
    """
    experiment_name: str = "g1_dwaq_amp"
    wandb_project: str = "g1_dwaq_amp"
    runner_class_name: str = "DWAQAMPOnPolicyRunner"

    # AMP-specific top-level parameters (required by DWAQAMPOnPolicyRunner)
    amp_motion_files: list = None
    amp_num_preload_transitions: int = 200000
    amp_reward_coef: float = 0.002
    amp_task_reward_lerp: float = 0.3
    amp_discr_hidden_dims: list = None

    def __post_init__(self):
        super().__post_init__()

        # ==================== DWAQ Settings (Inherited) ====================
        # Keep DWAQ policy and encoder architecture
        self.policy.class_name = "ActorCritic_DWAQ"
        self.policy.cenet_out_dim = 19  # velocity(3) + latent(16)

        # ==================== AMP Settings ====================
        # Use DWAQAMPPPO algorithm (combines DWAQ autoencoder + AMP discriminator)
        self.algorithm.class_name = "DWAQAMPPPO"

        # AMP Expert Motion Files
        # - 使用多个专家数据文件增强 AMP 效果
        # - 包含不同受试者的多种行走模式
        # - AMPLoader 会根据轨迹长度自动加权采样
        import os
        datasets_dir = "/home/c211/WorkSpace/G1DWAQ_Lab/TienKung-Lab/legged_lab/envs/g1/datasets"

        self.amp_motion_files = [
            os.path.join(datasets_dir, "run1_subject2.csv"),
            os.path.join(datasets_dir, "run1_subject5.csv"),
            os.path.join(datasets_dir, "run2_subject1.csv"),
            os.path.join(datasets_dir, "run2_subject4.csv"),
            # os.path.join(datasets_dir, "walk2_subject1.csv"),
            # os.path.join(datasets_dir, "walk2_subject3.csv"),
            # os.path.join(datasets_dir, "walk2_subject4.csv"),
            # os.path.join(datasets_dir, "walk3_subject1.csv"),
            # os.path.join(datasets_dir, "walk3_subject2.csv"),
            # os.path.join(datasets_dir, "walk3_subject3.csv"),
            # os.path.join(datasets_dir, "walk3_subject4.csv"),
            # os.path.join(datasets_dir, "walk3_subject5.csv"),
            # os.path.join(datasets_dir, "walk4_subject1.csv"),
        ]
        self.algorithm.amp_motion_files = self.amp_motion_files

        # AMP Reward Configuration
        # - amp_reward_coef: Weight of discriminator reward in total training
        #   Higher = stronger emphasis on matching expert motion
        #   Range: 0.001-0.01 (from original AMP paper)
        self.amp_reward_coef = 0.01
        self.algorithm.amp_reward_coef = self.amp_reward_coef

        # - amp_task_reward_lerp: Blend ratio between task and AMP reward
        #   total_reward = task_reward * (1-lerp) + amp_reward * lerp
        #   lerp=0.3: prioritize task (terrain adaptation) over style (natural gait)
        #   lerp=1.0: pure AMP training (only natural gait, ignoring task)
        self.amp_task_reward_lerp = 0.3
        self.algorithm.amp_task_reward_lerp = self.amp_task_reward_lerp

        # Motion Replay Buffer
        # - Preload expert transitions for discriminator training
        # - Original AMP uses 2M transitions for A1 robot
        # - 使用多个专家数据文件时，增加预加载数量以覆盖更多样性
        # - 12个文件 × ~50K transitions/file ≈ 600K transitions
        self.amp_num_preload_transitions = 1000000
        self.algorithm.amp_num_preload_transitions = self.amp_num_preload_transitions

        # Discriminator Network Architecture
        # - Input: state pairs (current_state + next_state)
        # - Output: expert score (higher = more expert-like)
        # - Architecture: [1024, 512, 256] (3-layer MLP)
        self.amp_discr_hidden_dims = [1024, 512, 256]
        self.algorithm.amp_discr_hidden_dims = self.amp_discr_hidden_dims

        # ==================== Training Hyperparameters ====================
        # Keep DWAQ's higher entropy for exploration
        self.algorithm.entropy_coef = 0.01

        # AMP typically uses longer replay buffer than DWAQ
        self.algorithm.amp_replay_buffer_size = 1000000
