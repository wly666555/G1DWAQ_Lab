# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class G1AmpOnPolicyRunner(AmpOnPolicyRunner):
    """
    G1 AMP On-Policy Runner.

    在 AmpOnPolicyRunner 基础上补全 G1AmpAgentCfg 中未设置的
    min_normalized_std（G1 有 29 个 DOF）。
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        if not train_cfg.get("min_normalized_std"):
            train_cfg["min_normalized_std"] = [0.05] * env.num_actions

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
