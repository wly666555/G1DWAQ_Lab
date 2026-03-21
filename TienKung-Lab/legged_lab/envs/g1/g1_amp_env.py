# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
G1 AMP Environment

纯 AMP（不含 DWAQ VAE encoder）的 G1 复杂地形行走环境。

继承 G1DwaqEnv，复用：
- 地形课程（ROUGH_TERRAINS_CFG）
- 高度扫描（actor 可见）
- 步态相位追踪
- 特权观测（critic 用）

新增 AMP 接口：
- get_amp_observations()             -> [num_envs, 58]
- get_amp_obs_for_expert_trans()     -> [num_envs, 58]  (AmpOnPolicyRunner 接口)
- reset_env_ids                      property

AMP 观测（58 维）：
- joint_pos (29) + joint_vel (29)
匹配 Male2Walking_c3d NPZ 数据格式。

step() 返回标准 4-tuple，与 AmpOnPolicyRunner 兼容：
  obs, rewards, dones, extras
  extras["observations"]["critic"] = privileged_obs
"""

from __future__ import annotations

import torch

from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.envs.g1.g1_amp_config import G1AmpEnvCfg


class G1AmpEnv(G1DwaqEnv):
    """
    G1 AMP environment — inherits all DWAQ terrain/scene logic,
    adds AMP observation interface for the discriminator.
    """

    def __init__(self, cfg: G1AmpEnvCfg, headless: bool):
        # AMP terminal states buffer（在 parent __init__ 调用 reset() 之前初始化）
        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.scene.num_envs

        self._terminal_amp_states = torch.zeros(
            self.num_envs,
            self._amp_obs_size(),
            device=self.device,
        )

        # reset_env_ids 用于 AmpOnPolicyRunner 获取 terminal states
        self._reset_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)

        super().__init__(cfg, headless)

    def _amp_obs_size(self) -> int:
        """AMP 观测维度：joint_pos(29) + joint_vel(29) = 58."""
        return 58

    def get_amp_observations(self) -> torch.Tensor:
        """
        返回当前 AMP 观测 [num_envs, 58]。

        格式与 Male2Walking_c3d NPZ 数据一致：
          dof_positions (29) + dof_velocities (29)
        """
        joint_pos = self.robot.data.joint_pos   # [num_envs, 29]
        joint_vel = self.robot.data.joint_vel   # [num_envs, 29]
        return torch.cat([joint_pos, joint_vel], dim=-1)

    def get_amp_obs_for_expert_trans(self) -> torch.Tensor:
        """
        AmpOnPolicyRunner 调用的接口（与原始 AMP runner 兼容）。
        等价于 get_amp_observations()。
        """
        return self.get_amp_observations()

    def reset(self, env_ids):
        """
        Reset 前保存 terminal AMP states，供 runner 使用。
        """
        if len(env_ids) == 0:
            return

        # 保存 terminal states（reset 之前的最后状态）
        self._terminal_amp_states[env_ids] = (
            self.get_amp_observations()[env_ids].clone()
        )
        self._reset_env_ids = env_ids.clone()

        super().reset(env_ids)

    def step(self, actions: torch.Tensor):
        """
        执行一步。

        AmpOnPolicyRunner 只需要 extras["observations"]["critic"]，
        移除 DWAQ 专用的 obs_hist / prev_critic_obs，避免无意义的内存开销。
        """
        obs, rewards, dones, extras = super().step(actions)

        extras["observations"] = {"critic": extras["observations"]["critic"]}
        return obs, rewards, dones, extras

    @property
    def reset_env_ids(self) -> torch.Tensor:
        """最近一次 reset 的环境 IDs（AmpOnPolicyRunner 读取 terminal states 用）。"""
        return self._reset_env_ids

    def get_observations(self):
        """
        覆盖 G1DwaqEnv.get_observations()，返回 AmpOnPolicyRunner 期望的格式：
          (obs, extras)
          extras["observations"]["critic"] = privileged_obs

        G1DwaqEnv 原版返回 (obs, obs_hist)，不符合 AmpOnPolicyRunner 接口。
        """
        actor_obs, critic_obs = self.compute_observations()

        self.extras["observations"] = {
            "critic": critic_obs,
        }
        return actor_obs, self.extras
