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
DWAQ + AMP On-Policy Runner

This runner combines:
1. DWAQ: Observation history for VAE encoder, autoencoder loss
2. AMP: Expert motion data, discriminator training, AMP reward

Training Loop:
- Collect rollouts with DWAQ actor (obs + latent code)
- Collect AMP observations for discriminator
- Update with combined loss: PPO + autoencoder + discriminator
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import DWAQAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic_DWAQ, Discriminator, EmpiricalNormalization
from rsl_rl.utils.g1_motion_loader import AMPLoader
from rsl_rl.utils import Normalizer, store_code_state


class DWAQAMPOnPolicyRunner:
    """On-policy runner for DWAQ + AMP training."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        """
        Initialize the DWAQ + AMP on-policy runner.

        Args:
            env: The vectorized environment with DWAQ and AMP interfaces.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for saving logs and checkpoints.
            device: Device to run training on ('cpu' or 'cuda:X').
        """
        self.cfg = train_cfg
        self.train_cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Configure multi-gpu
        self._configure_multi_gpu()

        # Resolve observation dimensions
        num_obs = self.env.num_obs
        num_privileged_obs = self.env.num_privileged_obs if self.env.num_privileged_obs is not None else num_obs
        num_obs_hist = self.env.num_obs_hist

        # Calculate encoder dimensions
        cenet_in_dim = num_obs_hist * num_obs
        cenet_out_dim = self.policy_cfg.get("cenet_out_dim", 19)

        # Create DWAQ policy
        policy_class_name = self.policy_cfg.get("class_name", "ActorCritic_DWAQ")
        policy_class = eval(policy_class_name)

        filtered_policy_cfg = {k: v for k, v in self.policy_cfg.items() if k in ["activation", "init_noise_std"]}

        policy: ActorCritic_DWAQ = policy_class(
            num_actor_obs=num_obs + cenet_out_dim,
            num_critic_obs=num_privileged_obs,
            num_actions=self.env.num_actions,
            cenet_in_dim=cenet_in_dim,
            cenet_out_dim=cenet_out_dim,
            obs_dim=num_obs,
            **filtered_policy_cfg,
        ).to(self.device)

        # Initialize AMP components
        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)

        # Initialize DWAQAMPPPO algorithm
        alg_class_name = self.alg_cfg.get("class_name", "DWAQAMPPPO")
        alg_class = eval(alg_class_name)

        dwaq_amp_alg_supported_params = [
            "num_learning_epochs", "num_mini_batches", "clip_param",
            "gamma", "lam", "value_loss_coef", "entropy_coef",
            "learning_rate", "max_grad_norm", "use_clipped_value_loss",
            "schedule", "desired_kl", "amp_replay_buffer_size",
        ]
        filtered_alg_cfg = {k: v for k, v in self.alg_cfg.items() if k in dwaq_amp_alg_supported_params}

        self.alg: DWAQAMPPPO = alg_class(
            policy=policy,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=self.device,
            obs_dim=num_obs,
            **filtered_alg_cfg,
        )

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_obs],
            critic_obs_shape=[num_privileged_obs],
            obs_hist_shape=[cenet_in_dim],
            action_shape=[self.env.num_actions],
        )

        # Initialize observation normalizers
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # Decide whether to disable logging
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging setup
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _configure_multi_gpu(self):
        """Configure multi-GPU training settings."""
        self.is_distributed = torch.distributed.is_initialized()
        if self.is_distributed:
            self.gpu_global_rank = torch.distributed.get_rank()
            self.gpu_world_size = torch.distributed.get_world_size()
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Run the training loop for DWAQ + AMP.

        Args:
            num_learning_iterations: Number of training iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        # Initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self._init_logger()

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, obs_hist = self.env.get_observations()
        privileged_obs, prev_critic_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs

        # Move to device
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        prev_critic_obs = prev_critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)
        amp_obs = amp_obs.to(self.device)

        # Switch to train mode
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced for multi-gpu
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout phase
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions using DWAQ + AMP actor
                    actions = self.alg.act(obs, critic_obs, prev_critic_obs, obs_hist, amp_obs)

                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Extract DWAQ-specific data from extras
                    obs_dict = extras.get("observations", {})
                    privileged_obs = obs_dict.get("critic", None)
                    obs_hist = obs_dict.get("obs_hist", obs_hist)
                    prev_critic_obs = obs_dict.get("prev_critic_obs", prev_critic_obs)
                    amp_obs = self.env.get_amp_observations()

                    # Update critic observations
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    # Move to device
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    prev_critic_obs = prev_critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    amp_obs = amp_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Process environment step
                    self.alg.process_env_step(rewards, dones, extras, amp_obs)

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # Compute returns
                self.alg.compute_returns(critic_obs)

            # Update phase
            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop

            mean_value_loss, mean_surrogate_loss, mean_autoenc_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = (
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )

            # Get beta value for VAE
            beta = self._get_beta_schedule(it)

            # Update algorithm
            loss_dict = self.alg.update(beta=beta)
            mean_value_loss = loss_dict.get("value_function", 0.0)
            mean_surrogate_loss = loss_dict.get("surrogate", 0.0)
            mean_autoenc_loss = loss_dict.get("autoencoder", 0.0)
            mean_amp_loss = loss_dict.get("amp", 0.0)
            mean_grad_pen_loss = loss_dict.get("amp_grad_pen", 0.0)
            mean_policy_pred = loss_dict.get("amp_policy_pred", 0.0)
            mean_expert_pred = loss_dict.get("amp_expert_pred", 0.0)

            stop = time.time()
            learn_time = stop - start

            # Log training info
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())

            # Save model
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Update iteration counter
            self.current_learning_iteration = it + 1
            ep_infos.clear()

        # Save final model
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _get_beta_schedule(self, iteration: int) -> float:
        """Get beta value for VAE KL divergence loss scheduling."""
        beta_schedule = self.cfg.get("beta_schedule", "constant")
        if beta_schedule == "constant":
            return 1.0
        elif beta_schedule == "linear":
            beta_start = self.cfg.get("beta_start", 0.0)
            beta_end = self.cfg.get("beta_end", 1.0)
            beta_steps = self.cfg.get("beta_steps", 1000)
            return min(beta_start + (beta_end - beta_start) * iteration / beta_steps, beta_end)
        else:
            return 1.0

    def _init_logger(self):
        """Initialize logger (Tensorboard, WandB, or Neptune)."""
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()

        if self.logger_type == "neptune":
            from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
        elif self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Store git status
        if self.log_dir is not None:
            store_code_state(self.log_dir, self.git_status_repos)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training statistics."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    val = ep_info[key].to(self.device)
                    # Handle scalar tensors by adding a dimension
                    if val.dim() == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/autoencoder", locs["mean_autoenc_loss"], locs["it"])
        self.writer.add_scalar("Loss/amp", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/amp_grad_pen", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Loss/amp_policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/amp_expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'Autoencoder loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
            f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
            f"""{'AMP grad penalty:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
            f"""{'AMP policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
            f"""{'AMP expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['tot_iter'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None):
        """Save model checkpoint."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None):
        """Get policy for inference."""
        self.alg.policy.eval()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy

    def train_mode(self):
        """Set policy to training mode."""
        self.alg.train_mode()

    def eval_mode(self):
        """Set policy to evaluation mode."""
        self.alg.test_mode()
