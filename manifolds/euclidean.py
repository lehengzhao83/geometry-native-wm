# manifolds/euclidean.py
# A minimal, robust Euclidean manifold implementation (R^d) with a clean interface.
# This is intended to be "hard to break" and to match the same API as other manifolds.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Euclidean:
    """
    Euclidean manifold R^d.

    Conventions:
      - Points x live in R^d (any last-dim size is allowed).
      - Tangent vectors v live in the same space as points.
      - exp_map(x, v) = x + v
      - log_map(x, y) = y - x
      - dist(x, y) = ||y - x||_2
    """

    eps: float = 1e-12

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Projection to the manifold (identity for Euclidean)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Euclidean.proj expects a torch.Tensor, got {type(x)}")
        return x

    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Projection to tangent space (identity for Euclidean)."""
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Euclidean.proj_tan expects a torch.Tensor, got {type(v)}")
        # x is unused; kept for API compatibility
        return v

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at x applied to tangent vector v."""
        self._check_same_shape(x, v, "exp_map")
        return x + v

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at x sending y to tangent space at x."""
        self._check_same_shape(x, y, "log_map")
        return y - x

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Riemannian inner product at x (standard dot product)."""
        self._check_same_shape(u, v, "inner")
        # x is unused; kept for API compatibility
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def norm(self, x: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Riemannian norm of tangent vector v at x."""
        # x is unused; kept for API compatibility
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Euclidean.norm expects v as torch.Tensor, got {type(v)}")
        n = torch.linalg.norm(v, dim=-1, keepdim=keepdim)
        return n

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Geodesic distance between x and y."""
        self._check_same_shape(x, y, "dist")
        d = torch.linalg.norm(y - x, dim=-1, keepdim=keepdim)
        # guard against tiny negative from weird dtypes (rare)
        return torch.clamp(d, min=0.0)

    def squared_dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Squared distance between x and y."""
        self._check_same_shape(x, y, "squared_dist")
        diff = y - x
        d2 = (diff * diff).sum(dim=-1, keepdim=keepdim)
        return torch.clamp(d2, min=0.0)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Provided for API-compatibility with hyperbolic code; in Euclidean space it is simple addition.
        """
        self._check_same_shape(x, y, "mobius_add")
        return x + y

    def random(self, *shape: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Sample random points from standard normal in R^d."""
        if len(shape) == 0:
            raise ValueError("Euclidean.random requires a shape, e.g., (batch, dim)")
        return torch.randn(*shape, device=device, dtype=dtype)

    @staticmethod
    def _check_same_shape(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"Euclidean.{where} expects torch.Tensor inputs.")
        if a.shape != b.shape:
            raise ValueError(
                f"Euclidean.{where} shape mismatch: a.shape={tuple(a.shape)} vs b.shape={tuple(b.shape)}"
            )


# A small alias many projects like:
EuclideanManifold = Euclidean

