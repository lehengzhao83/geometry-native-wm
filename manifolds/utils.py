# manifolds/utils.py
# Shared utilities for manifold implementations.
#
# Goals:
# - "Hard to break" helpers for:
#   - safe dtype/device casting
#   - numerical clamps
#   - stable norms
#   - deterministic seeding helpers
# - Optional factory to build manifolds from a simple config dict.
#
# This file has NO heavy dependencies beyond torch.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import os
import random

import numpy as np
import torch


# ---------------------------
# Seeding / determinism
# ---------------------------
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make CUDA deterministic as much as possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For some ops in recent PyTorch versions
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ---------------------------
# Numeric helpers
# ---------------------------
def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    Stable L2 norm with clamping.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"safe_norm expects torch.Tensor, got {type(x)}")
    n = torch.linalg.norm(x, dim=dim, keepdim=keepdim)
    return torch.clamp(n, min=eps)


def clamp_min(x: torch.Tensor, min_val: float) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"clamp_min expects torch.Tensor, got {type(x)}")
    return torch.clamp(x, min=min_val)


def clamp_max(x: torch.Tensor, max_val: float) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"clamp_max expects torch.Tensor, got {type(x)}")
    return torch.clamp(x, max=max_val)


def to_device_dtype(
    x: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"to_device_dtype expects torch.Tensor, got {type(x)}")
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x


def infer_device(device: str = "auto") -> torch.device:
    """
    device: "auto" | "cpu" | "cuda"
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)
    raise ValueError(f"Unknown device spec: {device}")


def infer_dtype(dtype: str = "float32") -> torch.dtype:
    m = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if dtype not in m:
        raise ValueError(f"Unknown dtype spec: {dtype}. Valid: {list(m.keys())}")
    return m[dtype]


# ---------------------------
# Simple manifold factory (optional but useful)
# ---------------------------
@dataclass
class ManifoldBundle:
    """
    Convenience wrapper that stores:
    - manifold object
    - dim (last dimension)
    - name
    """
    name: str
    manifold: Any
    dim: int


def build_manifold_from_cfg(cfg: Dict[str, Any]) -> ManifoldBundle:
    """
    Build a manifold from a simple config dict.

    Expected patterns:
      {"name": "euclid", "type": "euclidean", "dim": 64}
      {"name": "hyp", "type": "hyperbolic", "dim": 64, "curvature": 1.0}
      {"name": "circle", "type": "circle", "num_circles": 8, "representation": "unit_vector_2d"}

    Returns a ManifoldBundle(name, manifold, dim)

    NOTE:
    - For Circle with unit_vector_2d, dim = 2*num_circles.
    - For Circle with angle, dim = num_circles.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"build_manifold_from_cfg expects dict, got {type(cfg)}")
    if "type" not in cfg:
        raise ValueError("Manifold cfg must include 'type'")
    mtype = str(cfg["type"]).lower()
    name = str(cfg.get("name", mtype))

    if mtype in ("euclidean", "euclid", "r"):
        from .euclidean import Euclidean

        dim = int(cfg["dim"])
        return ManifoldBundle(name=name, manifold=Euclidean(eps=float(cfg.get("eps", 1e-12))), dim=dim)

    if mtype in ("hyperbolic", "poincare", "poincare_ball", "h"):
        from .hyperbolic import PoincareBall

        dim = int(cfg["dim"])
        c = float(cfg.get("curvature", cfg.get("c", 1.0)))
        eps = float(cfg.get("eps", 1e-5))
        return ManifoldBundle(name=name, manifold=PoincareBall(c=c, eps=eps), dim=dim)

    if mtype in ("circle", "s1", "torus"):
        from .circle import Circle

        num_circles = int(cfg.get("num_circles", 1))
        rep = str(cfg.get("representation", "unit_vector_2d"))
        eps = float(cfg.get("eps", 1e-6))
        man = Circle(num_circles=num_circles, representation=rep, eps=eps)
        dim = man.dim
        return ManifoldBundle(name=name, manifold=man, dim=dim)

    raise ValueError(f"Unknown manifold type: {mtype}")


def build_product_from_cfg(product_cfg: Dict[str, Any]):
    """
    Build ProductManifold from a config dict like:
      {"factors": [
          {"name":"periodic", "type":"circle", "num_circles":8, "representation":"unit_vector_2d"},
          {"name":"euclid", "type":"euclidean", "dim":64}
      ]}
    """
    if not isinstance(product_cfg, dict) or "factors" not in product_cfg:
        raise ValueError("product_cfg must be a dict with key 'factors'")

    from .product import FactorSpec, ProductManifold

    factors_cfg = product_cfg["factors"]
    if not isinstance(factors_cfg, list) or len(factors_cfg) == 0:
        raise ValueError("'factors' must be a non-empty list")

    specs = []
    for fcfg in factors_cfg:
        bundle = build_manifold_from_cfg(fcfg)
        specs.append(FactorSpec(name=bundle.name, manifold=bundle.manifold, dim=bundle.dim))

    return ProductManifold(specs)

