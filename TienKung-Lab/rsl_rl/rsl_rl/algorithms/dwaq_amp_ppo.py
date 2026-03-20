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
DWAQ + AMP PPO Algorithm

This algorithm combines:
1. DWAQ (Deep Variational Autoencoder for Walking): β-VAE for terrain adaptation
2. AMP (Adversarial Motion Priors): Discriminator for natural gait learning

Training Loss:
total_loss = ppo_loss + autoencoder_loss + amp_discriminator_loss

Components:
- PPO: Surrogate loss + value loss + entropy
- DWAQ: Velocity prediction + reconstruction + KL divergence
- AMP: Expert/policy discrimination + gradient penalty
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic_DWAQ
from rsl_rl.storage import RolloutStorageDWAQ, ReplayBuffer


class DWAQAMPPPO:
    """Proximal Policy Optimization with DWAQ context encoder and AMP discriminator.

    This algorithm extends DWAQPPO by adding AMP's adversarial motion prior learning.
    The discriminator learns to distinguish between expert and policy motions, providing
    an additional reward signal for natural gait.

    References:
    - DWAQ: DreamWaQ (https://github.com/Gepetto/DreamWaQ)
    - AMP: Adversarial Motion Priors (https://arxiv.org/abs/2104.02180)
    """

    policy: ActorCritic_DWAQ
    """The actor critic module with DWAQ context encoder."""

    def __init__(
        self,
        policy: ActorCritic_DWAQ,
        discriminator,
        amp_data,
        amp_normalizer,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        obs_dim: int = 45,
        amp_replay_buffer_size: int = 100000,
    ):
        """Initialize the DWAQ + AMP PPO algorithm.

        Args:
            policy: The actor-critic network with DWAQ context encoder.
            discriminator: AMP discriminator network.
            amp_data: Expert motion data replay buffer.
            amp_normalizer: Normalizer for AMP observations.
            num_learning_epochs: Number of learning epochs per update.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda parameter.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            learning_rate: Learning rate for optimizer.
            max_grad_norm: Maximum gradient norm for clipping.
            use_clipped_value_loss: Whether to use clipped value loss.
            schedule: Learning rate schedule ("fixed" or "adaptive").
            desired_kl: Desired KL divergence for adaptive schedule.
            device: Device to run on.
            obs_dim: Observation dimension (without latent code).
            amp_replay_buffer_size: Size of AMP replay buffer.
        """
        # Device configuration
        self.device = device
        self.obs_dim = obs_dim

        # Learning rate schedule parameters
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO + DWAQ components
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorageDWAQ | None = None  # initialized later
        self.transition = RolloutStorageDWAQ.Transition()

        # AMP components
        self.amploss_coef = 1.0
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorageDWAQ.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Optimizer for policy + discriminator
        params = [
            {"params": self.policy.parameters(), "name": "policy"},
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        obs_hist_shape: list[int],
        action_shape: list[int],
    ) -> None:
        """Initialize the rollout storage.

        Args:
            num_envs: Number of parallel environments.
            num_transitions_per_env: Number of transitions per environment per update.
            actor_obs_shape: Shape of actor observations.
            critic_obs_shape: Shape of critic observations (privileged).
            obs_hist_shape: Shape of observation history for context encoder.
            action_shape: Shape of actions.
        """
        self.storage = RolloutStorageDWAQ(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_hist_shape,
            action_shape,
            self.device,
        )

    def test_mode(self) -> None:
        """Set the policy to evaluation mode."""
        self.policy.eval()

    def train_mode(self) -> None:
        """Set the policy to training mode."""
        self.policy.train()

    def broadcast_parameters(self) -> None:
        """Broadcast parameters to all processes for multi-GPU training."""
        if torch.distributed.is_initialized():
            for param in self.policy.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        prev_critic_obs: torch.Tensor,
        obs_history: torch.Tensor,
        amp_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute actions for the given observations.

        Args:
            obs: Current actor observations.
            critic_obs: Current critic observations (privileged).
            prev_critic_obs: Previous critic observations for velocity target.
            obs_history: Observation history for context encoder.
            amp_obs: AMP observations for discriminator.

        Returns:
            Actions to execute in the environment.
        """
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs, obs_history).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # Record observations before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs
        self.transition.prev_critic_obs = prev_critic_obs

        # Record AMP observations
        self.amp_transition.observations = amp_obs

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        amp_obs: torch.Tensor,
    ) -> None:
        """Process the environment step results.

        Args:
            rewards: Rewards from the environment.
            dones: Done flags from the environment.
            infos: Additional info from the environment.
            amp_obs: AMP observations for next state.
        """
        # Record rewards and dones
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition for PPO
        self.storage.add_transitions(self.transition)

        # Record the transition for AMP discriminator
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # Clear transitions
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_critic_obs: Last critic observations for bootstrapping.
        """
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, beta: float = 1.0) -> dict[str, float]:
        """Perform a DWAQ + AMP PPO update step.

        Args:
            beta: Weight for the VAE KL divergence loss (β-VAE).

        Returns:
            Dictionary containing the mean losses.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_autoenc_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        # Generator for mini batches
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        # AMP generators
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                critic_obs_batch,
                prev_critic_obs_batch,
                obs_hist_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            # Recompute actions log prob and values for current batch
            self.policy.act(
                obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive learning rate based on KL divergence
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ==================== DWAQ: β-VAE loss for context encoder ====================
            (
                code,
                code_vel,
                decode,
                mean_vel,
                logvar_vel,
                mean_latent,
                logvar_latent,
            ) = self.policy.cenet_forward(obs_hist_batch)

            # Velocity target from current privileged observations
            vel_target = critic_obs_batch[:, self.obs_dim : self.obs_dim + 3].detach()
            decode_target = obs_batch[:, :self.obs_dim]
            vel_target.requires_grad = False
            decode_target.requires_grad = False

            # Autoencoder loss: velocity + reconstruction + KL divergence
            logvar_latent_clamped = torch.clamp(logvar_latent, min=-10.0, max=10.0)
            kl_divergence = -0.5 * torch.sum(
                1 + logvar_latent_clamped - mean_latent.pow(2) - logvar_latent_clamped.exp()
            )
            autoenc_loss = (
                nn.MSELoss()(code_vel, vel_target)
                + nn.MSELoss()(decode, decode_target)
                + beta * kl_divergence
            ) / self.num_mini_batches

            # ==================== PPO: Surrogate loss ====================
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ==================== PPO: Value function loss ====================
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ==================== AMP: Discriminator loss ====================
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # Normalize AMP observations
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            # Discriminator predictions
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            # Discriminator loss: expert should be 1, policy should be -1
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # Gradient penalty for discriminator
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

            # ==================== Total loss ====================
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + autoenc_loss
                + self.amploss_coef * amp_loss
                + self.amploss_coef * grad_pen_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update AMP normalizer
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_autoenc_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates

        # Clear storage
        self.storage.clear()

        # Construct loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "autoencoder": mean_autoenc_loss,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }

        return loss_dict
