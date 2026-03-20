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
#
# This file contains code derived from the RSL-RL, Isaac Lab, Legged Lab,
# DreamWaQ, and AMP Projects, with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
G1 DWAQ + AMP Environment

This environment combines:
1. DWAQ: Blind walking with VAE encoder for terrain adaptation
2. AMP: Adversarial motion priors for natural gait learning

Key Features:
- Inherits all DWAQ functionality (observation history, VAE encoder)
- Adds AMP observations for discriminator training
- AMP observations format matches the expert data from AMP_for_g1 project

AMP Observations (65 dimensions):
- joint_pos: 29 dimensions (all joint positions)
- base_lin_vel: 3 dimensions (root linear velocity in world frame)
- base_ang_vel: 3 dimensions (root angular velocity in body frame)
- joint_vel: 29 dimensions (all joint velocities)
- z_pos: 1 dimension (root height)

Total: 29 + 3 + 3 + 29 + 1 = 65 dimensions

Reference: AMP_for_g1/legged_gym/envs/base/g1_legged_robot.py::get_amp_observations()
"""

from __future__ import annotations

import torch

from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.envs.g1.g1_dwaq_amp_config import G1DwaqAmpEnvCfg


class G1DwaqAmpEnv(G1DwaqEnv):
    """
    G1 DWAQ + AMP environment for natural blind walking.

    Extends G1DwaqEnv with AMP observations for discriminator training.
    The AMP observations are used to train a discriminator that distinguishes
    between policy-generated motions and expert motions from mocap data.
    """

    def __init__(
        self,
        cfg: G1DwaqAmpEnvCfg,
        headless: bool,
    ):
        """
        Initialize G1 DWAQ + AMP environment.

        Args:
            cfg: Environment configuration (G1DwaqAmpEnvCfg)
            headless: Whether to run in headless mode
        """
        # Store config and device before parent init
        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.scene.num_envs

        # AMP-specific: Initialize terminal AMP states BEFORE parent init
        # This is required because parent __init__ calls reset() which needs this buffer
        self.terminal_amp_states = torch.zeros(
            self.num_envs,
            self._get_amp_obs_size(),
            device=self.device
        )

        # Initialize parent DWAQ environment
        super().__init__(cfg, headless)

    def _get_amp_obs_size(self) -> int:
        """
        Get the size of AMP observations.

        Returns:
            Size of AMP observations (58 dimensions for G1)
        """
        # AMP obs: joint_pos(29) + joint_vel(29) = 58 dimensions
        # This matches the NPZ expert data format (dof_positions + dof_velocities)
        return 29 + 29

    def get_amp_observations(self) -> torch.Tensor:
        """
        Get AMP observations for discriminator training.

        This method returns the state representation used by the AMP discriminator
        to distinguish between policy-generated and expert motions.

        The format matches the NPZ expert data format:
        - joint_pos: All 29 joint positions (absolute, not relative to default)
        - joint_vel: All 29 joint velocities

        This simplified representation (58D) matches the structure of the Male2Walking_c3d
        NPZ dataset which contains dof_positions (29) + dof_velocities (29).

        Returns:
            AMP observations [num_envs, 58]

        Note: Previous version used 65D with base velocities and root height, but the
        NPZ expert data only contains joint-level information.
        """
        robot = self.robot

        # Joint positions (29 dimensions) - use absolute positions, not relative to default
        joint_pos = robot.data.joint_pos  # [num_envs, 29]

        # Joint velocities (29 dimensions)
        joint_vel = robot.data.joint_vel  # [num_envs, 29]

        # Concatenate to match NPZ format: dof_positions + dof_velocities
        amp_obs = torch.cat([
            joint_pos,      # 29
            joint_vel,      # 29
        ], dim=-1)  # Total: 58 dimensions

        return amp_obs

    def reset(self, env_ids):
        """
        Reset specified environments.

        Extends parent reset to also store terminal AMP states for reset environments.
        This is required for AMP discriminator training to have correct terminal states.

        Args:
            env_ids: Environment IDs to reset
        """
        if len(env_ids) == 0:
            return

        # Store terminal AMP states BEFORE resetting
        # These are used by the discriminator as the "next state" for terminal transitions
        self.terminal_amp_states[env_ids] = self.get_amp_observations()[env_ids].clone()

        # Call parent reset
        super().reset(env_ids)

    def step(self, actions: torch.Tensor):
        """
        Execute one environment step.

        Extends parent step to include terminal AMP states in extras.

        Returns:
            obs: Actor observations [num_envs, num_obs]
            rewards: Reward values [num_envs]
            dones: Done flags [num_envs]
            extras: Dict containing:
                - observations.critic: Privileged observations
                - observations.obs_hist: DWAQ observation history
                - observations.prev_critic_obs: Previous critic obs
                - terminal_amp_states: Terminal AMP states for reset envs
                - time_outs: Timeout flags
                - log: Episode statistics
        """
        # Call parent step
        obs, rewards, dones, extras = super().step(actions)

        # Add terminal AMP states to extras for discriminator training
        # The runner will use these to provide correct next_state for terminal transitions
        extras["terminal_amp_states"] = self.terminal_amp_states

        return obs, rewards, dones, extras

    def get_amp_observations_for_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Get terminal AMP observations for reset environments.

        This is called by the runner to get the terminal state for environments
        that are being reset. The terminal state is used as the "next state"
        for the last transition before reset in discriminator training.

        Args:
            env_ids: Environment IDs that are being reset

        Returns:
            Terminal AMP observations for specified environments [len(env_ids), 65]
        """
        return self.terminal_amp_states[env_ids]
