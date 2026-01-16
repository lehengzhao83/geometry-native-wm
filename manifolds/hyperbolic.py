# manifolds/hyperbolic.py
# Robust Hyperbolic (Poincaré Ball) manifold implementation in pure PyTorch.
#
# Guarantees:
# - No dependency on geoopt/geomstats (works even if they are missing).
# - Numerically stable with projection to the ball.
# - Supports curvature c > 0 (sectional curvature = -c).
#
# Reference formulas (standard Poincaré ball model):
# - Ball radius R = 1/sqrt(c)
# - Möbius addition ⊕_c
# - Exp/Log maps
# - Distance d_c(x,y) = (2/sqrt(c)) * artanh( sqrt(c) * || (-x) ⊕_c y || )
#
# Conventions:
# - Points x are tensors with last dim = d (any leading batch dims).
# - Tangent vectors live in R^d (same shape as points).
# - All operations broadcast over leading dims.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


def _artanh(x: torch.Tensor) -> torch.Tensor:
    # Stable artanh for |x| < 1
    # artanh(x) = 0.5 * (log(1+x) - log(1-x))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass
class PoincareBall:
    """
    Poincaré ball model with curvature c > 0 (i.e., sectional curvature -c).

    Parameters
    ----------
    c : float
        Positive curvature parameter (negative curvature is -c).
    eps : float
        Numerical stability epsilon.
    """

    c: float = 1.0
    eps: float = 1e-5

    # ---------------------------
    # Basic helpers
    # ---------------------------
    def _check_c(self) -> float:
        if not (isinstance(self.c, (float, int))):
            raise TypeError(f"PoincareBall.c must be float/int, got {type(self.c)}")
        if self.c <= 0:
            raise ValueError(f"PoincareBall.c must be > 0, got {self.c}")
        return float(self.c)

    @property
    def radius(self) -> float:
        c = self._check_c()
        return 1.0 / (c ** 0.5)

    def _norm(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return torch.linalg.norm(x, dim=-1, keepdim=keepdim)

    def _lambda_x(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        # Conformal factor: λ_x = 2 / (1 - c||x||^2)
        c = self._check_c()
        x2 = (x * x).sum(dim=-1, keepdim=keepdim)
        denom = 1.0 - c * x2
        denom = torch.clamp(denom, min=self.eps)
        return 2.0 / denom

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points into the open ball of radius 1/sqrt(c).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"PoincareBall.proj expects torch.Tensor, got {type(x)}")
        c = self._check_c()
        r = 1.0 / (c ** 0.5)

        # Max norm slightly below radius
        maxnorm = (1.0 - self.eps) * r
        norm = self._norm(x, keepdim=True)
        # Scale down if norm >= maxnorm
        cond = norm >= maxnorm
        scale = maxnorm / torch.clamp(norm, min=self.eps)
        return torch.where(cond, x * scale, x)

    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Tangent space at x is R^d (same ambient), so identity.
        Kept for API compatibility.
        """
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"PoincareBall.proj_tan expects torch.Tensor v, got {type(v)}")
        # x unused but kept for consistent signature
        return v

    # ---------------------------
    # Möbius operations
    # ---------------------------
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition on the Poincaré ball:
        x ⊕_c y = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
        """
        self._check_same_shape(x, y, "mobius_add")
        c = self._check_c()
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        denom = 1.0 + 2.0 * c * xy + (c * c) * x2 * y2
        denom = torch.clamp(denom, min=self.eps)
        out = num / denom
        return self.proj(out)

    def mobius_neg(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"PoincareBall.mobius_neg expects torch.Tensor, got {type(x)}")
        return -x

    # ---------------------------
    # Exp / Log maps
    # ---------------------------
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at x applied to tangent vector v.
        exp_x^c(v) = x ⊕_c ( tanh( sqrt(c) * λ_x * ||v|| / 2 ) * v / (sqrt(c) * ||v||) )
        """
        self._check_same_shape(x, v, "exp_map")
        c = self._check_c()
        x = self.proj(x)

        v_norm = self._norm(v, keepdim=True)
        v_norm = torch.clamp(v_norm, min=self.eps)

        lam = self._lambda_x(x, keepdim=True)  # λ_x
        sqrt_c = c ** 0.5

        # factor = tanh( sqrt(c) * λ_x * ||v|| / 2 ) / (sqrt(c) * ||v||)
        theta = (sqrt_c * lam * v_norm) / 2.0
        factor = torch.tanh(theta) / (sqrt_c * v_norm)
        second = v * factor
        out = self.mobius_add(x, second)
        return self.proj(out)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at x sending y to tangent space at x.
        log_x^c(y) = (2 / (sqrt(c) * λ_x)) * artanh( sqrt(c) * || (-x)⊕_c y || ) * u
        where u = ((-x)⊕_c y) / ||(-x)⊕_c y||
        """
        self._check_same_shape(x, y, "log_map")
        c = self._check_c()
        x = self.proj(x)
        y = self.proj(y)

        sub = self.mobius_add(self.mobius_neg(x), y)  # (-x) ⊕ y
        sub_norm = self._norm(sub, keepdim=True)
        sub_norm = torch.clamp(sub_norm, min=self.eps)

        lam = self._lambda_x(x, keepdim=True)
        sqrt_c = c ** 0.5

        # artanh argument: sqrt(c) * ||sub||
        arg = torch.clamp(sqrt_c * sub_norm, max=1.0 - self.eps)
        scale = (2.0 / (sqrt_c * lam)) * _artanh(arg) / sub_norm
        v = sub * scale
        return v

    # ---------------------------
    # Distance / inner product / norms
    # ---------------------------
    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Geodesic distance:
        d_c(x,y) = (2/sqrt(c)) * artanh( sqrt(c) * || (-x)⊕_c y || )
        """
        self._check_same_shape(x, y, "dist")
        c = self._check_c()
        x = self.proj(x)
        y = self.proj(y)

        sub = self.mobius_add(self.mobius_neg(x), y)
        sub_norm = self._norm(sub, keepdim=True)

        sqrt_c = c ** 0.5
        arg = torch.clamp(sqrt_c * sub_norm, max=1.0 - self.eps)
        d = (2.0 / sqrt_c) * _artanh(arg)
        if not keepdim:
            d = d.squeeze(-1)
        return torch.clamp(d, min=0.0)

    def squared_dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        d = self.dist(x, y, keepdim=True)
        d2 = d * d
        if not keepdim:
            d2 = d2.squeeze(-1)
        return d2

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Riemannian inner product at x:
        <u,v>_x = (λ_x^2 / 4) * <u,v>_E
        """
        self._check_same_shape(u, v, "inner")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"PoincareBall.inner expects torch.Tensor x, got {type(x)}")
        x = self.proj(x)
        lam = self._lambda_x(x, keepdim=True)
        eu = (u * v).sum(dim=-1, keepdim=True)
        ip = (lam * lam / 4.0) * eu
        if not keepdim:
            ip = ip.squeeze(-1)
        return ip

    def norm(self, x: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Riemannian norm at x:
        ||v||_x = (λ_x / 2) * ||v||_E
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"PoincareBall.norm expects torch.Tensor x, got {type(x)}")
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"PoincareBall.norm expects torch.Tensor v, got {type(v)}")
        x = self.proj(x)
        lam = self._lambda_x(x, keepdim=True)
        n = (lam / 2.0) * torch.linalg.norm(v, dim=-1, keepdim=True)
        if not keepdim:
            n = n.squeeze(-1)
        return n

    # ---------------------------
    # Sampling
    # ---------------------------
    def random(
        self,
        *shape: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        std: float = 0.1,
    ) -> torch.Tensor:
        """
        Sample points by drawing from a small Gaussian then projecting into the ball.
        std should be small to avoid hitting the boundary too often.
        """
        if len(shape) == 0:
            raise ValueError("PoincareBall.random requires a shape, e.g., (batch, dim)")
        x = torch.randn(*shape, device=device, dtype=dtype) * float(std)
        return self.proj(x)

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _check_same_shape(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"PoincareBall.{where} expects torch.Tensor inputs.")
        if a.shape != b.shape:
            raise ValueError(
                f"PoincareBall.{where} shape mismatch: a.shape={tuple(a.shape)} vs b.shape={tuple(b.shape)}"
            )


# Common alias
HyperbolicPoincare = PoincareBall

