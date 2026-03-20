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
    amp_task_reward_lerp: float = 0.2
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

        # AMP Expert Motion Files - NPZ Format
        # - 使用多个 npz 格式的专家数据文件增强 AMP 效果
        # - 自动加载 Male2Walking_c3d 目录下所有 .npz 文件
        # - AMPLoaderNPZ 会合并所有文件的帧数据用于采样
        import os

        # NPZ 专家数据目录
        npz_datasets_dir = "/home/c211/WorkSpace/G1DWAQ_Lab/TienKung-Lab/legged_lab/envs/g1/datasets/Male2Walking_c3d"

        # 自动枚举所有 .npz 文件
        if os.path.isdir(npz_datasets_dir):
            npz_files = [
                os.path.join(npz_datasets_dir, f)
                for f in os.listdir(npz_datasets_dir)
                if f.endswith('.npz')
            ]
            npz_files.sort()  # 排序确保一致性
            self.amp_motion_files = npz_files
            print(f"[G1DwaqAmpConfig] Found {len(npz_files)} npz files in {npz_datasets_dir}")
        else:
            print(f"[WARNING] NPZ directory not found: {npz_datasets_dir}")
            self.amp_motion_files = []

        self.algorithm.amp_motion_files = self.amp_motion_files

        # AMP Reward Configuration
        # - amp_reward_coef: Weight of discriminator reward in total training
        #   Higher = stronger emphasis on matching expert motion
        #   Range: 0.001-0.01 (from original AMP paper)
        #
        #   针对复杂地形行走（台阶）的调整:
        #   - 行走数据提供自然步态先验
        #   - 在保证地形适应的同时，学习自然的上台阶动作
        #   - 建议: 0.005-0.01 范围（不要太大，避免影响地形任务）
        self.amp_reward_coef = 0.008  # 适中配置，平衡自然性和地形适应
        self.algorithm.amp_reward_coef = self.amp_reward_coef

        # - amp_task_reward_lerp: Blend ratio between task and AMP reward
        #   total_reward = task_reward * (1-lerp) + amp_reward * lerp
        #
        #   针对复杂地形行走任务的调整:
        #   - 主要目标: 地形适应（上台阶、走斜坡等）
        #   - 次要目标: 步态自然性
        #   - 建议: 0.2-0.3 范围（20-30% AMP，70-80% 任务）
        #   - 当前: 0.25 (75% 任务, 25% AMP) - 优先保证地形穿越能力
        self.amp_task_reward_lerp = 0.2  # 地形任务优先，辅以自然步态学习
        self.algorithm.amp_task_reward_lerp = self.amp_task_reward_lerp

        # Motion Replay Buffer
        # - Preload expert transitions for discriminator training
        # - Original AMP uses 2M transitions for A1 robot
        #
        #   针对 Male2Walking_c3d 行走数据的调整:
        #   - 23 个 NPZ 文件，总共约 3734 帧行走数据
        #   - 这些数据提供平地自然行走的先验
        #   - 判别器学习自然步态特征，应用到台阶等复杂地形
        #   - 设置为 10K 足够覆盖所有专家数据的多样性
        self.amp_num_preload_transitions = 10000  # 使用所有可用帧（实际约3734）
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
