# datasets/toy_pose.py
# Toy Pose World dataset (WORKS OUT-OF-THE-BOX, no external assets)
#
# Purpose:
# - Provide a "pose / rotation" sequential task to validate geometric state spaces.
# - We start with SO(2) which is exactly S^1, so it works with Circle manifold immediately.
# - Optionally add Euclidean nuisance factors (translation + scale) and occlusions/noise.
#
# What it generates:
# - A simple binary shape (square / triangle / circle) rendered into a 64x64 grayscale image
# - The shape undergoes rotation theta(t) with constant angular velocity (per-sequence)
# - Optional translation and scale changes
#
# Returns per sample:
#   {
#     "x":      FloatTensor [T, 1, H, W]   # frames in [0,1]
#     "theta":  FloatTensor [T, 1]         # wrapped angle in (-pi,pi]
#     "omega":  FloatTensor [1]            # angular velocity
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


def _make_base_grid(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a normalized coordinate grid in [-1,1]x[-1,1]
    Returns:
      X, Y: [H,W]
    """
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return X, Y


def _render_shape_mask(
    shape: str,
    X: np.ndarray,
    Y: np.ndarray,
    theta: float,
    tx: float,
    ty: float,
    scale: float,
) -> np.ndarray:
    """
    Render a shape mask by inverse-transforming coordinates (rotate+translate+scale)
    and evaluating an implicit shape equation.

    Coordinates X,Y are normalized in [-1,1].
    tx,ty in normalized coords.
    """
    # inverse translation
    x = (X - tx) / max(scale, 1e-6)
    y = (Y - ty) / max(scale, 1e-6)

    # inverse rotation
    c = np.cos(-theta).astype(np.float32)
    s = np.sin(-theta).astype(np.float32)
    xr = c * x - s * y
    yr = s * x + c * y

    shape = shape.lower()
    if shape == "circle":
        r = np.sqrt(xr * xr + yr * yr)
        mask = (r <= 0.45).astype(np.float32)
        return mask
    if shape == "square":
        mask = ((np.abs(xr) <= 0.45) & (np.abs(yr) <= 0.45)).astype(np.float32)
        return mask
    if shape == "triangle":
        # upright triangle in local coords: vertices approx at (0,0.55), (-0.5,-0.35), (0.5,-0.35)
        # Use barycentric test with half-plane inequalities.
        x0, y0 = 0.0, 0.55
        x1, y1 = -0.5, -0.35
        x2, y2 = 0.5, -0.35

        # Compute sign of area for each edge
        def edge(px, py, ax, ay, bx, by):
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        e0 = edge(xr, yr, x0, y0, x1, y1)
        e1 = edge(xr, yr, x1, y1, x2, y2)
        e2 = edge(xr, yr, x2, y2, x0, y0)

        # Ensure consistent orientation (triangle is CCW here)
        mask = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)).astype(np.float32)
        return mask

    raise ValueError(f"Unknown shape: {shape}")


def _soft_edges(mask: np.ndarray, blur: int = 1) -> np.ndarray:
    """
    Simple blur to make gradients nicer (not required, but helps learning).
    blur=0 disables.
    """
    if blur <= 0:
        return mask
    m = mask
    for _ in range(blur):
        # 3x3 average filter (manual, numpy)
        p = np.pad(m, ((1, 1), (1, 1)), mode="edge")
        m = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return m.astype(np.float32)


class ToyPoseDataset(Dataset):
    """
    SO(2) rotation toy dataset rendered as images.
    """

    def __init__(
        self,
        size: int,
        seq_len: int = 16,
        image_size: int = 64,
        shapes: Tuple[str, ...] = ("circle", "square", "triangle"),
        # angle dynamics
        omega_min: float = 0.10,        # rad/step
        omega_max: float = 0.50,        # rad/step
        # nuisance factors
        translate_enabled: bool = True,
        tx_range: float = 0.25,         # normalized coords
        ty_range: float = 0.25,
        scale_enabled: bool = True,
        scale_min: float = 0.85,
        scale_max: float = 1.15,
        # rendering noise/occlusion
        noise_std: float = 0.02,
        occlusion_enabled: bool = True,
        p_occlude: float = 0.10,
        occluder_size_min: int = 8,
        occluder_size_max: int = 16,
        blur: int = 1,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be positive int, got {size}")
        if not isinstance(seq_len, int) or seq_len <= 1:
            raise ValueError(f"seq_len must be int > 1, got {seq_len}")
        if not isinstance(image_size, int) or image_size <= 8:
            raise ValueError(f"image_size must be int > 8, got {image_size}")
        if omega_min <= 0 or omega_max <= 0 or omega_max < omega_min:
            raise ValueError("omega_min/omega_max must be >0 and omega_max>=omega_min")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 <= float(p_occlude) < 1.0):
            raise ValueError("p_occlude must be in [0,1)")

        self.size = size
        self.seq_len = seq_len
        self.H = image_size
        self.W = image_size
        self.shapes = tuple(shapes)

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        self.translate_enabled = bool(translate_enabled)
        self.tx_range = float(tx_range)
        self.ty_range = float(ty_range)

        self.scale_enabled = bool(scale_enabled)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        self.noise_std = float(noise_std)
        self.occlusion_enabled = bool(occlusion_enabled)
        self.p_occlude = float(p_occlude)
        self.occl_min = int(occluder_size_min)
        self.occl_max = int(occluder_size_max)
        self.blur = int(blur)

        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

        # Precompute base grid
        self.X, self.Y = _make_base_grid(self.H, self.W)

    def __len__(self) -> int:
        return self.size

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(idx) * 10007)

    def _apply_occlusion(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.occlusion_enabled or rng.random() >= self.p_occlude:
            return img
        H, W = img.shape
        s = int(rng.integers(self.occl_min, self.occl_max + 1))
        top = int(rng.integers(0, max(1, H - s + 1)))
        left = int(rng.integers(0, max(1, W - s + 1)))
        img2 = img.copy()
        img2[top : top + s, left : left + s] = 0.0
        return img2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._rng_for_index(idx)
        T = self.seq_len

        shape = self.shapes[int(rng.integers(0, len(self.shapes)))]

        theta0 = float(rng.uniform(-np.pi, np.pi))
        omega = float(rng.uniform(self.omega_min, self.omega_max))
        # random direction
        if rng.random() < 0.5:
            omega = -omega

        # nuisance initial conditions
        if self.translate_enabled:
            tx = float(rng.uniform(-self.tx_range, self.tx_range))
            ty = float(rng.uniform(-self.ty_range, self.ty_range))
        else:
            tx, ty = 0.0, 0.0

        if self.scale_enabled:
            scale = float(rng.uniform(self.scale_min, self.scale_max))
        else:
            scale = 1.0

        frames = np.zeros((T, 1, self.H, self.W), dtype=np.float32)
        thetas = np.zeros((T, 1), dtype=np.float32)

        for t in range(T):
            th = theta0 + omega * t
            th = float(_wrap_to_pi(np.array(th, dtype=np.float32)))
            thetas[t, 0] = th

            mask = _render_shape_mask(shape, self.X, self.Y, th, tx, ty, scale)
            mask = _soft_edges(mask, blur=self.blur)

            # add noise and occlusion
            if self.noise_std > 0:
                mask = mask + rng.standard_normal(mask.shape).astype(np.float32) * self.noise_std
            mask = np.clip(mask, 0.0, 1.0)
            mask = self._apply_occlusion(mask, rng)

            frames[t, 0] = mask

        x = torch.from_numpy(frames).to(dtype=self.dtype)
        theta = torch.from_numpy(thetas).to(dtype=self.dtype)
        omega_t = torch.tensor([omega], dtype=self.dtype)

        if self.device is not None:
            x = x.to(self.device)
            theta = theta.to(self.device)
            omega_t = omega_t.to(self.device)

        return {"x": x, "theta": theta, "omega": omega_t}


def build_toy_pose_datasets(cfg: Dict) -> Dict[str, ToyPoseDataset]:
    """
    Build datasets dict: {"train":..., "val":..., "test":..., "test_ood":... (optional)}.

    Expects cfg like a toy_pose.yaml (not provided here), but you can reuse patterns from real_video.yaml.
    """
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))

    split_cfg = data_cfg.get("split", {})
    scene_cfg = data_cfg.get("scene", {})
    dyn_cfg = data_cfg.get("dynamics", {})
    nuis_cfg = data_cfg.get("nuisance", {})
    rend_cfg = data_cfg.get("render", {})

    common = dict(
        seq_len=int(data_cfg.get("sequence", {}).get("length", 16)),
        image_size=int(scene_cfg.get("image_size", 64)),
        shapes=tuple(scene_cfg.get("shapes", ["circle", "square", "triangle"])),
        omega_min=float(dyn_cfg.get("omega", {}).get("min", 0.10)),
        omega_max=float(dyn_cfg.get("omega", {}).get("max", 0.50)),
        translate_enabled=bool(nuis_cfg.get("translate", {}).get("enabled", True)),
        tx_range=float(nuis_cfg.get("translate", {}).get("tx_range", 0.25)),
        ty_range=float(nuis_cfg.get("translate", {}).get("ty_range", 0.25)),
        scale_enabled=bool(nuis_cfg.get("scale", {}).get("enabled", True)),
        scale_min=float(nuis_cfg.get("scale", {}).get("min", 0.85)),
        scale_max=float(nuis_cfg.get("scale", {}).get("max", 1.15)),
        noise_std=float(rend_cfg.get("noise_std", 0.02)),
        occlusion_enabled=bool(rend_cfg.get("occlusion", {}).get("enabled", True)),
        p_occlude=float(rend_cfg.get("occlusion", {}).get("p_occlude", 0.10)),
        occluder_size_min=int(rend_cfg.get("occlusion", {}).get("size_min", 8)),
        occluder_size_max=int(rend_cfg.get("occlusion", {}).get("size_max", 16)),
        blur=int(rend_cfg.get("blur", 1)),
        seed=seed,
        device=None,
        dtype=torch.float32,
    )

    datasets: Dict[str, ToyPoseDataset] = {
        "train": ToyPoseDataset(size=int(split_cfg.get("train_size", 20000)), **common),
        "val": ToyPoseDataset(size=int(split_cfg.get("val_size", 2000)), **common),
        "test": ToyPoseDataset(size=int(split_cfg.get("test_size", 2000)), **common),
    }

    ood_cfg = data_cfg.get("ood", {})
    if bool(ood_cfg.get("enabled", False)):
        common_ood = dict(common)
        common_ood.update(
            omega_min=float(ood_cfg.get("test_omega", {}).get("min", common["omega_min"])),
            omega_max=float(ood_cfg.get("test_omega", {}).get("max", common["omega_max"])),
            noise_std=float(ood_cfg.get("test_noise_std", common["noise_std"])),
            p_occlude=float(ood_cfg.get("test_p_occlude", common["p_occlude"])),
            seed=seed + 99991,
        )
        datasets["test_ood"] = ToyPoseDataset(size=int(split_cfg.get("test_size", 2000)), **common_ood)

    return datasets

