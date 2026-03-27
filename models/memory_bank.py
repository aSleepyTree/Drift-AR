"""
Class-wise ring buffer memory bank for drifting training.
Ported from the official JAX implementation: https://github.com/lambertae/drifting/blob/main/memory_bank.py
"""

import numpy as np
import torch
from typing import Optional, Tuple


class ArrayMemoryBank:
    """Per-class ring buffer that stores feature/latent samples for drift loss."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros(
            (self.num_classes, self.max_size, *self.feature_shape),
            dtype=self.dtype,
        )

    def add(self, samples, labels) -> None:
        """Insert samples into per-class ring buffers.

        Args:
            samples: tensor or array of shape (N, *feature_shape).
            labels:  integer class labels of shape (N,).
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        samples = np.asarray(samples, dtype=self.dtype)
        labels = np.asarray(labels).astype(np.int64)

        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            if lbl < 0 or lbl >= self.num_classes:
                continue
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int, device="cuda") -> torch.Tensor:
        """Sample stored entries for each label.

        Args:
            labels:    integer class labels of shape (B,).
            n_samples: number of samples to draw per label.
            device:    target torch device.

        Returns:
            torch.Tensor of shape (B, n_samples, *feature_shape).
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels).astype(np.int64)

        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)

        for i in range(bsz):
            lbl = int(labels[i])
            if lbl < 0 or lbl >= self.num_classes:
                lbl = 0
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros(n_samples, dtype=np.int32)
            else:
                sample_indices[i] = np.random.choice(
                    valid, n_samples, replace=(valid < n_samples)
                )

        out = self.bank[labels[:, None], sample_indices]
        return torch.from_numpy(out).to(device=device)

    def is_ready(self, min_count: int = 1) -> bool:
        """Check if the bank has at least min_count entries in every class."""
        if self.bank is None:
            return False
        return bool(np.all(self.count >= min_count))

    def total_count(self) -> int:
        return int(self.count.sum())
