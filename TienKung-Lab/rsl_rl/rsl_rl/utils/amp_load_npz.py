# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
AMP Loader for NPZ Format Expert Motion Data

This loader supports loading multiple .npz files containing expert motion data
for Adversarial Motion Priors (AMP) training.

NPZ File Format:
- Each .npz file should contain:
  - 'positions': numpy array of shape (num_frames, num_features)
  - 'velocities': numpy array of shape (num_frames, num_features)

The loader concatenates positions and velocities to form the full state representation,
and merges data from multiple files for diverse expert motion samples.
"""

import os
import numpy as np
import torch


class AMPLoaderNPZ:
    """
    AMP Motion Loader for NPZ format expert data.

    Loads multiple .npz files and concatenates all frames for sampling.
    Each npz file should contain 'positions' and 'velocities' arrays.
    """

    def __init__(self, npz_files: list[str], device: str = "cpu", num_preload_transitions: int = 1000000):
        """
        Initialize AMP loader for npz files.

        Args:
            npz_files: List of paths to .npz files
            device: Device to store tensors ('cpu' or 'cuda')
            num_preload_transitions: Maximum number of transitions to preload
        """
        self.device = device
        self.npz_files = npz_files
        self.frame_data = None
        self.num_frames = 0
        self.num_preload_transitions = num_preload_transitions

        print(f"[AMPLoaderNPZ] Loading {len(npz_files)} npz files...")
        self.load_npz_files(npz_files)

    def load_npz_files(self, filepaths: list[str]):
        """
        Load and concatenate data from multiple npz files.

        Args:
            filepaths: List of npz file paths
        """
        all_frames = []
        total_frames_before_limit = 0

        for i, filepath in enumerate(filepaths):
            assert os.path.isfile(filepath), f"NPZ file not found: {filepath}"

            print(f"[AMPLoaderNPZ] Loading file {i+1}/{len(filepaths)}: {os.path.basename(filepath)}")

            data = np.load(filepath)

            # Check required keys - support both naming conventions
            if 'dof_positions' in data and 'dof_velocities' in data:
                positions = data['dof_positions']  # Shape: (num_frames, num_joints)
                velocities = data['dof_velocities']  # Shape: (num_frames, num_joints)
            elif 'positions' in data and 'velocities' in data:
                positions = data['positions']
                velocities = data['velocities']
            else:
                print(f"[WARNING] File {filepath} missing position/velocity data, skipping")
                print(f"  Available keys: {list(data.keys())}")
                continue

            assert positions.shape[0] == velocities.shape[0], \
                f"Frame count mismatch in {filepath}: positions={positions.shape[0]}, velocities={velocities.shape[0]}"

            # Concatenate positions and velocities to form full state
            frames = np.concatenate([positions, velocities], axis=1)  # Shape: (num_frames, 2*num_joints)

            total_frames_before_limit += frames.shape[0]
            all_frames.append(frames)

            print(f"  - Loaded {frames.shape[0]} frames, feature dim: {frames.shape[1]}")

        # Concatenate all frames from all files
        self.frame_data = np.concatenate(all_frames, axis=0)  # Shape: (total_frames, feature_dim)

        # Limit to num_preload_transitions if specified
        if self.num_preload_transitions > 0 and self.frame_data.shape[0] > self.num_preload_transitions:
            print(f"[AMPLoaderNPZ] Limiting transitions from {self.frame_data.shape[0]} to {self.num_preload_transitions}")
            indices = np.random.choice(self.frame_data.shape[0], self.num_preload_transitions, replace=False)
            self.frame_data = self.frame_data[indices]

        self.num_frames = self.frame_data.shape[0]
        self.observation_dim = self.frame_data.shape[1]  # AmpOnPolicyRunner 需要此属性

        print(f"[AMPLoaderNPZ] Successfully loaded {self.num_frames} total frames from {len(self.npz_files)} files")
        print(f"[AMPLoaderNPZ] Feature dimension: {self.observation_dim}")

    def sample(self, batch_size: int):
        """
        Sample random frames from expert data.

        Args:
            batch_size: Number of frames to sample

        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        indices = np.random.choice(self.num_frames, batch_size, replace=True)
        batch = self.frame_data[indices]
        return torch.tensor(batch, dtype=torch.float32, device=self.device)

    def sample_state_pairs(self, batch_size: int):
        """
        Sample consecutive state pairs (current_state, next_state) for discriminator training.

        Args:
            batch_size: Number of state pairs to sample

        Returns:
            Tuple of (current_states, next_states), each of shape (batch_size, feature_dim)
        """
        # Sample indices, ensuring we don't sample the last frame
        idx = np.random.choice(self.num_frames - 1, batch_size, replace=True)
        curr_states = self.frame_data[idx]
        next_states = self.frame_data[idx + 1]

        return (
            torch.tensor(curr_states, dtype=torch.float32, device=self.device),
            torch.tensor(next_states, dtype=torch.float32, device=self.device)
        )

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """
        Generator for feeding expert data in mini-batches.
        Compatible with AMP training loop.

        Args:
            num_mini_batch: Number of mini-batches to generate
            mini_batch_size: Size of each mini-batch

        Yields:
            Tuple of (current_states, next_states) for each mini-batch
        """
        for _ in range(num_mini_batch):
            yield self.sample_state_pairs(mini_batch_size)
