# datasets/toy_periodic.py
# Toy Periodic World dataset (WORKS OUT-OF-THE-BOX)
#
# What it generates:
# - K independent periodic factors (torus) with per-sequence random frequency & amplitude
# - Observations as concatenated (sin, cos) for each factor, then optionally linearly mixed
# - Optional missingness masks (Bernoulli or block)
#
# Designed to match configs/toy_periodic.yaml.
#
# Returns (per sample):
#   sample = {
#     "x":     FloatTensor [T, obs_dim]     # observations
#     "mask":  FloatTensor [T, obs_dim]     # 1 for observed, 0 for missing (if enabled)
#     "theta": FloatTensor [T, K]           # ground-truth phases in (-pi, pi]
#   }
#
# Dependencies: torch, numpy

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    # Random orthogonal via QR
    A = rng.standard_normal((n, n)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Fix sign for deterministic-ish
    sign = np.sign(np.diag(R))
    Q = Q * sign
    return Q.astype(np.float32)


@dataclass
class MixMatrix:
    W: np.ndarray  # [in_dim, out_dim]


class ToyPeriodicDataset(Dataset):
    """
    Synthetic periodic sequences.

    Each sample:
      K = num_sines periodic factors
      theta_k(t) = theta0_k + 2π f_k * t * dt
      signal_k(t) = [A_k sin(theta_k(t)), A_k cos(theta_k(t))]

    Then optionally mixed:
      x(t) = concat_k signal_k(t)  -> [T, 2K]
      y(t) = x(t) @ W              -> [T, obs_dim] (if obs_dim != 2K or mixing enabled)

    Missingness:
      mask(t, :) is 0 for missing steps (set y(t,:) = 0 to keep shapes stable).
    """

    def __init__(
        self,
        size: int,
        seq_len: int,
        num_sines: int = 3,
        dt: float = 1.0,
        freq_min: float = 0.03,
        freq_max: float = 0.15,
        amp_min: float = 0.5,
        amp_max: float = 1.5,
        obs_dim: int = 16,
        mix_matrix: str = "random_orthogonal",  # random_orthogonal | random_gaussian | identity
        noise_std: float = 0.05,
        # missingness
        missing_enabled: bool = True,
        p_missing: float = 0.15,
        missing_mode: str = "bernoulli",  # bernoulli | block
        block_min_len: int = 2,
        block_max_len: int = 6,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be positive int, got {size}")
        if not isinstance(seq_len, int) or seq_len <= 1:
            raise ValueError(f"seq_len must be int > 1, got {seq_len}")
        if not isinstance(num_sines, int) or num_sines <= 0:
            raise ValueError(f"num_sines must be positive int, got {num_sines}")
        if freq_min <= 0 or freq_max <= 0 or freq_max < freq_min:
            raise ValueError("freq_min/freq_max must be >0 and freq_max>=freq_min")
        if amp_min <= 0 or amp_max <= 0 or amp_max < amp_min:
            raise ValueError("amp_min/amp_max must be >0 and amp_max>=amp_min")
        if not isinstance(obs_dim, int) or obs_dim <= 0:
            raise ValueError("obs_dim must be positive int")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if missing_mode not in ("bernoulli", "block"):
            raise ValueError(f"missing_mode must be bernoulli|block, got {missing_mode}")
        if not (0.0 <= float(p_missing) < 1.0):
            raise ValueError("p_missing must be in [0,1)")
        if block_min_len <= 0 or block_max_len < block_min_len:
            raise ValueError("invalid block length range")

        self.size = size
        self.seq_len = seq_len
        self.K = num_sines
        self.dt = float(dt)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.amp_min = float(amp_min)
        self.amp_max = float(amp_max)
        self.obs_dim = obs_dim
        self.mix_matrix_type = str(mix_matrix).lower()
        self.noise_std = float(noise_std)

        self.missing_enabled = bool(missing_enabled)
        self.p_missing = float(p_missing)
        self.missing_mode = str(missing_mode).lower()
        self.block_min_len = int(block_min_len)
        self.block_max_len = int(block_max_len)

        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

        # Build fixed mixing matrix (shared across all samples to keep task consistent)
        in_dim = 2 * self.K
        rng = np.random.default_rng(self.seed)

        if self.mix_matrix_type == "identity":
            W = np.eye(in_dim, obs_dim, dtype=np.float32)
        elif self.mix_matrix_type == "random_gaussian":
            W = rng.standard_normal((in_dim, obs_dim)).astype(np.float32)
            # scale for stability
            W = W / max(1.0, np.linalg.norm(W, axis=0, keepdims=True).mean())
        elif self.mix_matrix_type == "random_orthogonal":
            # If obs_dim != in_dim, take first obs_dim columns of orthogonal matrix
            n = max(in_dim, obs_dim)
            Q = _random_orthogonal(rng, n)
            W = Q[:in_dim, :obs_dim].astype(np.float32)
        else:
            raise ValueError(f"Unknown mix_matrix: {mix_matrix}")

        self.W = torch.from_numpy(W)  # [2K, obs_dim]

        # Precompute time indices
        self.t_idx = np.arange(self.seq_len, dtype=np.float32) * self.dt

    def __len__(self) -> int:
        return self.size

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(idx) * 10007)

    def _sample_missing_steps(self, rng: np.random.Generator) -> np.ndarray:
        """
        Returns boolean array [T] where True means missing.
        """
        T = self.seq_len
        miss = np.zeros((T,), dtype=np.bool_)
        if not self.missing_enabled or self.p_missing <= 0:
            return miss

        if self.missing_mode == "bernoulli":
            miss = rng.random((T,)) < self.p_missing
            return miss.astype(np.bool_)

        # block missingness
        # number of blocks based on expected missing fraction
        expected_missing = int(round(T * self.p_missing))
        if expected_missing <= 0:
            return miss
        # greedily place blocks
        remaining = expected_missing
        attempts = 0
        while remaining > 0 and attempts < 10 * T:
            attempts += 1
            L = int(rng.integers(self.block_min_len, self.block_max_len + 1))
            L = min(L, remaining)
            start = int(rng.integers(0, max(1, T - L + 1)))
            miss[start : start + L] = True
            remaining = expected_missing - int(miss.sum())
        return miss

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng_for_index(idx)
        T = self.seq_len
        K = self.K

        # Sample per-sequence params
        freq = rng.uniform(self.freq_min, self.freq_max, size=(K,)).astype(np.float32)  # cycles per step
        amp = rng.uniform(self.amp_min, self.amp_max, size=(K,)).astype(np.float32)
        theta0 = rng.uniform(-np.pi, np.pi, size=(K,)).astype(np.float32)

        # theta(t) = theta0 + 2pi * f * t
        theta = theta0[None, :] + (2.0 * np.pi) * (self.t_idx[:, None] * freq[None, :])
        theta = _wrap_to_pi(theta).astype(np.float32)  # [T,K]

        sinv = np.sin(theta) * amp[None, :]
        cosv = np.cos(theta) * amp[None, :]
        x_raw = np.concatenate([sinv, cosv], axis=1).astype(np.float32)  # [T, 2K]

        x = torch.from_numpy(x_raw)  # [T,2K]
        W = self.W.to(dtype=self.dtype)
        x = x.to(dtype=self.dtype) @ W  # [T, obs_dim]

        # Add observation noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * float(self.noise_std)

        # Missingness
        miss = self._sample_missing_steps(rng)  # [T] bool
        if self.missing_enabled:
            mask_t = (~miss).astype(np.float32)[:, None]  # [T,1]
            mask = torch.from_numpy(mask_t).to(dtype=self.dtype).repeat(1, self.obs_dim)  # [T,obs_dim]
            x = x * mask
        else:
            mask = torch.ones((T, self.obs_dim), dtype=self.dtype)

        theta_t = torch.from_numpy(theta).to(dtype=self.dtype)  # [T,K]

        if self.device is not None:
            x = x.to(self.device)
            mask = mask.to(self.device)
            theta_t = theta_t.to(self.device)

        return {"x": x, "mask": mask, "theta": theta_t}


def build_toy_periodic_datasets(cfg: Dict) -> Dict[str, ToyPeriodicDataset]:
    """
    Build datasets dict: {"train":..., "val":..., "test":..., "test_ood":... (optional)}.

    Expects cfg like configs/toy_periodic.yaml loaded into a dict.
    """
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))

    seq_cfg = data_cfg.get("sequence", {})
    obs_cfg = data_cfg.get("observation", {})
    miss_cfg = data_cfg.get("missingness", {})
    split_cfg = data_cfg.get("split", {})

    common = dict(
        seq_len=int(seq_cfg.get("length", 32)),
        num_sines=int(seq_cfg.get("num_sines", 3)),
        dt=float(seq_cfg.get("dt", 1.0)),
        freq_min=float(seq_cfg.get("freq", {}).get("min", 0.03)),
        freq_max=float(seq_cfg.get("freq", {}).get("max", 0.15)),
        amp_min=float(seq_cfg.get("amp", {}).get("min", 0.5)),
        amp_max=float(seq_cfg.get("amp", {}).get("max", 1.5)),
        obs_dim=int(obs_cfg.get("obs_dim", 16)),
        mix_matrix=str(obs_cfg.get("mix_matrix", "random_orthogonal")),
        noise_std=float(obs_cfg.get("noise_std", 0.05)),
        missing_enabled=bool(miss_cfg.get("enabled", True)),
        p_missing=float(miss_cfg.get("p_missing", 0.15)),
        missing_mode=str(miss_cfg.get("mode", "bernoulli")),
        block_min_len=int(miss_cfg.get("block", {}).get("min_len", 2)),
        block_max_len=int(miss_cfg.get("block", {}).get("max_len", 6)),
        seed=seed,
        device=None,
        dtype=torch.float32,
    )

    datasets: Dict[str, ToyPeriodicDataset] = {
        "train": ToyPeriodicDataset(size=int(split_cfg.get("train_size", 30000)), **common),
        "val": ToyPeriodicDataset(size=int(split_cfg.get("val_size", 3000)), **common),
        "test": ToyPeriodicDataset(size=int(split_cfg.get("test_size", 3000)), **common),
    }

    ood_cfg = data_cfg.get("ood", {})
    if bool(ood_cfg.get("enabled", False)):
        common_ood = dict(common)
        common_ood.update(
            freq_min=float(ood_cfg.get("test_freq", {}).get("min", common["freq_min"])),
            freq_max=float(ood_cfg.get("test_freq", {}).get("max", common["freq_max"])),
            amp_min=float(ood_cfg.get("test_amp", {}).get("min", common["amp_min"])),
            amp_max=float(ood_cfg.get("test_amp", {}).get("max", common["amp_max"])),
            noise_std=float(ood_cfg.get("test_noise_std", common["noise_std"])),
            p_missing=float(ood_cfg.get("test_p_missing", common["p_missing"])),
            seed=seed + 99991,
        )
        datasets["test_ood"] = ToyPeriodicDataset(size=int(split_cfg.get("test_size", 3000)), **common_ood)

    return datasets
