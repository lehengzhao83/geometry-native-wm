# manifolds/circle.py
# Robust S^1 (circle) manifold implementation in pure PyTorch.
#
# Supports two representations:
#   1) "unit_vector_2d": each circle component is represented as (cos θ, sin θ) on the unit circle.
#      - This is the MOST robust representation (no angle wrap issues).
#      - dim per circle = 2
#
#   2) "angle": each circle component is represented as θ in (-pi, pi].
#      - This is compact but requires careful wrapping; provided for completeness.
#      - dim per circle = 1
#
# This file implements a PRODUCT of circles (S^1)^k using the chosen representation.
#
# Interface matches other manifolds:
#   proj, proj_tan, exp_map, log_map, dist, squared_dist, inner, norm, random

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


def _wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    # Wrap to (-pi, pi]
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


@dataclass
class Circle:
    """
    Product of circles (S^1)^k.

    Parameters
    ----------
    num_circles : int
        Number of independent circle factors.
    representation : str
        "unit_vector_2d" or "angle"
    eps : float
        Numerical epsilon for projection and stability.
    """

    num_circles: int = 1
    representation: str = "unit_vector_2d"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.num_circles, int) or self.num_circles <= 0:
            raise ValueError(f"Circle.num_circles must be a positive int, got {self.num_circles}")
        if self.representation not in ("unit_vector_2d", "angle"):
            raise ValueError(
                f"Circle.representation must be 'unit_vector_2d' or 'angle', got {self.representation}"
            )

    @property
    def dim(self) -> int:
        # Ambient dimension for one point
        if self.representation == "unit_vector_2d":
            return 2 * self.num_circles
        return 1 * self.num_circles

    # ---------------------------
    # Projection
    # ---------------------------
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto (S^1)^k.

        unit_vector_2d: normalize each (cos,sin) pair
        angle: wrap to (-pi, pi]
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Circle.proj expects torch.Tensor, got {type(x)}")

        if x.shape[-1] != self.dim:
            raise ValueError(f"Circle.proj expects last dim={self.dim}, got {x.shape[-1]}")

        if self.representation == "unit_vector_2d":
            pairs = x.view(*x.shape[:-1], self.num_circles, 2)
            norms = torch.linalg.norm(pairs, dim=-1, keepdim=True).clamp_min(self.eps)
            pairs = pairs / norms
            return pairs.view(*x.shape[:-1], self.dim)

        # angle
        return _wrap_to_pi(x)

    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project v to the tangent space at x.

        unit_vector_2d: for each circle, tangent space at point p is vectors orthogonal to p.
                        v_tan = v - <v,p> p
        angle: tangent space is R^k, so identity.
        """
        if not isinstance(x, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("Circle.proj_tan expects torch.Tensor inputs.")
        if x.shape != v.shape:
            raise ValueError(f"Circle.proj_tan shape mismatch: x{tuple(x.shape)} vs v{tuple(v.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Circle.proj_tan expects last dim={self.dim}, got {x.shape[-1]}")

        x = self.proj(x)

        if self.representation == "unit_vector_2d":
            p = x.view(*x.shape[:-1], self.num_circles, 2)
            vv = v.view(*v.shape[:-1], self.num_circles, 2)
            # subtract component along p
            dot = (vv * p).sum(dim=-1, keepdim=True)
            vv_tan = vv - dot * p
            return vv_tan.view(*v.shape[:-1], self.dim)

        # angle
        return v

    # ---------------------------
    # Exp / Log maps
    # ---------------------------
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on (S^1)^k.

        unit_vector_2d:
            For each circle point p in R^2 (||p||=1) and tangent v (p·v=0):
            exp_p(v) = cos(||v||) p + sin(||v||) v/||v||
        angle:
            exp_x(v) = wrap(x + v)
        """
        self._check_same_shape(x, v, "exp_map")
        x = self.proj(x)

        if self.representation == "angle":
            return self.proj(x + v)

        # unit_vector_2d
        p = x.view(*x.shape[:-1], self.num_circles, 2)
        vv = self.proj_tan(x, v).view(*v.shape[:-1], self.num_circles, 2)

        vn = torch.linalg.norm(vv, dim=-1, keepdim=True).clamp_min(self.eps)  # (..., k, 1)
        # cos/sin of angle magnitude
        c = torch.cos(vn)
        s = torch.sin(vn)
        out = c * p + s * (vv / vn)
        out = out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(self.eps)
        return out.view(*x.shape[:-1], self.dim)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Log map on (S^1)^k.

        angle:
            log_x(y) = wrap(y - x) in (-pi,pi]
        unit_vector_2d:
            For each circle, with points p,q on S^1 in R^2:
              angle = atan2(det(p,q), dot(p,q)) in (-pi,pi]
              tangent direction u = R90(p) (a unit tangent)
              log_p(q) = angle * u
            where R90([a,b]) = [-b,a].
        """
        self._check_same_shape(x, y, "log_map")
        x = self.proj(x)
        y = self.proj(y)

        if self.representation == "angle":
            return self.proj(y - x)  # wrap difference

        p = x.view(*x.shape[:-1], self.num_circles, 2)
        q = y.view(*y.shape[:-1], self.num_circles, 2)

        # dot and det per circle
        dot = (p * q).sum(dim=-1, keepdim=True)  # (..., k, 1)
        det = (p[..., 0:1] * q[..., 1:2]) - (p[..., 1:2] * q[..., 0:1])  # (..., k, 1)
        angle = torch.atan2(det, dot)  # (..., k, 1) in (-pi, pi]

        # unit tangent at p: rotate p by +90 degrees
        u = torch.stack([-p[..., 1], p[..., 0]], dim=-1)  # (..., k, 2)
        # Ensure u is unit length (should be if p is unit)
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(self.eps)

        v = angle * u  # (..., k, 2)
        return v.view(*x.shape[:-1], self.dim)

    # ---------------------------
    # Distance / inner / norm
    # ---------------------------
    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Geodesic distance on (S^1)^k:
          d(x,y) = sqrt( sum_k angle_k^2 )
        where angle_k is the shortest angular difference for each circle.
        """
        self._check_same_shape(x, y, "dist")
        x = self.proj(x)
        y = self.proj(y)

        if self.representation == "angle":
            delta = _wrap_to_pi(y - x)  # (..., k)
            d = torch.linalg.norm(delta, dim=-1, keepdim=keepdim)
            return torch.clamp(d, min=0.0)

        # unit_vector_2d: use log_map magnitude per circle
        v = self.log_map(x, y).view(*x.shape[:-1], self.num_circles, 2)
        angle = torch.linalg.norm(v, dim=-1)  # (..., k)
        d = torch.linalg.norm(angle, dim=-1, keepdim=keepdim)
        return torch.clamp(d, min=0.0)

    def squared_dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        d = self.dist(x, y, keepdim=True)
        d2 = d * d
        if not keepdim:
            d2 = d2.squeeze(-1)
        return d2

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Riemannian inner product on (S^1)^k:
        It's the standard Euclidean inner product on each tangent space.

        unit_vector_2d: tangents are in R^2 per circle (orthogonal to p), so dot works.
        angle: tangents in R^k, so dot works.
        """
        self._check_same_shape(u, v, "inner")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Circle.inner expects torch.Tensor x, got {type(x)}")

        # project tangents for safety
        u = self.proj_tan(x, u)
        v = self.proj_tan(x, v)

        ip = (u * v).sum(dim=-1, keepdim=keepdim)
        return ip

    def norm(self, x: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Tangent norm (standard L2 norm over the ambient coordinates).
        """
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Circle.norm expects torch.Tensor v, got {type(v)}")
        v = self.proj_tan(x, v)
        n = torch.linalg.norm(v, dim=-1, keepdim=keepdim)
        return n

    # ---------------------------
    # Sampling
    # ---------------------------
    def random(
        self,
        *shape: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Sample random points uniformly (approximately) on each circle.

        If representation == unit_vector_2d:
          sample θ ~ U(-pi, pi), return (cosθ, sinθ) per circle.
        If representation == angle:
          sample θ ~ U(-pi, pi) per circle.
        """
        if len(shape) == 0:
            raise ValueError("Circle.random requires a shape prefix, e.g. (batch,)")
        # shape describes leading batch dims; last dim is implied by manifold
        batch_shape = shape
        theta = (torch.rand(*batch_shape, self.num_circles, device=device, dtype=dtype) * 2 * torch.pi) - torch.pi

        if self.representation == "angle":
            return theta.view(*batch_shape, self.dim)

        # unit_vector_2d
        cs = torch.cos(theta)
        sn = torch.sin(theta)
        out = torch.stack([cs, sn], dim=-1)  # (..., k, 2)
        return out.view(*batch_shape, self.dim)

    # ---------------------------
    # Utils
    # ---------------------------
    @staticmethod
    def _check_same_shape(a: torch.Tensor, b: torch.Tensor, where: str) -> None:
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"Circle.{where} expects torch.Tensor inputs.")
        if a.shape != b.shape:
            raise ValueError(f"Circle.{where} shape mismatch: a.shape={tuple(a.shape)} vs b.shape={tuple(b.shape)}")

