# manifolds/product.py
# Robust Product Manifold implementation in pure PyTorch.
#
# This class composes multiple manifold factors (Euclidean / Hyperbolic / Circle, etc.)
# into a single "product manifold" with a unified interface:
#   proj, proj_tan, exp_map, log_map, dist, squared_dist, inner, norm, random
#
# Points are represented as concatenation along the last dimension:
#   z = concat([z^(1), z^(2), ..., z^(K)], dim=-1)
#
# Each factor manifold is responsible for its own last-dim size.
#
# This file is intentionally defensive:
# - validates dims
# - stable concatenation/splitting
# - consistent keepdim behavior

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch


def _as_int(x, name: str) -> int:
    if not isinstance(x, int):
        raise TypeError(f"{name} must be int, got {type(x)}")
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")
    return x


@dataclass
class FactorSpec:
    """
    Specification for one factor in a product manifold.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. "periodic", "euclid", "hyperbolic").
    manifold : object
        Manifold instance providing required methods.
    dim : int
        The last-dimension size for this factor in the concatenated representation.
    """

    name: str
    manifold: object
    dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("FactorSpec.name must be a non-empty string")
        _as_int(self.dim, "FactorSpec.dim")


class ProductManifold:
    """
    Product manifold M = M1 x M2 x ... x MK.

    Requires each factor manifold to implement:
      proj(x), proj_tan(x, v), exp_map(x, v), log_map(x, y),
      dist(x, y, keepdim=False), squared_dist(x, y, keepdim=False) (optional),
      inner(x, u, v, keepdim=False), norm(x, v, keepdim=False), random(*shape)

    If a factor manifold does not implement squared_dist, ProductManifold will compute dist^2.

    Notes
    -----
    - For dist: we use the standard product metric:
        d(z, w) = sqrt( sum_k d_k(z_k, w_k)^2 )
      This is the most common and is what you want for the "geodesic_rollout" loss.
    """

    def __init__(self, factors: Sequence[FactorSpec], eps: float = 1e-12):
        if not isinstance(factors, (list, tuple)) or len(factors) == 0:
            raise ValueError("ProductManifold requires a non-empty list of FactorSpec factors.")
        self.factors: List[FactorSpec] = list(factors)
        self.eps = float(eps)

        # Validate factor dims and uniqueness of names Purely for debugging convenience
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError(f"ProductManifold factor names must be unique, got {names}")

        # Precompute slices for splitting/concatenation
        offsets = []
        start = 0
        for f in self.factors:
            end = start + int(f.dim)
            offsets.append((start, end))
            start = end
        self._slices: List[Tuple[int, int]] = offsets
        self.dim_total: int = start
        if self.dim_total <= 0:
            raise ValueError("ProductManifold total dim must be > 0")

    @property
    def dim(self) -> int:
        return self.dim_total

    # ---------------------------
    # Split / concat
    # ---------------------------
    def split(self, z: torch.Tensor) -> List[torch.Tensor]:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"ProductManifold.split expects torch.Tensor, got {type(z)}")
        if z.shape[-1] != self.dim_total:
            raise ValueError(f"ProductManifold.split expects last dim={self.dim_total}, got {z.shape[-1]}")
        parts: List[torch.Tensor] = []
        for (s, e), f in zip(self._slices, self.factors):
            part = z[..., s:e]
            if part.shape[-1] != f.dim:
                raise RuntimeError("Internal slice dim mismatch (should never happen).")
            parts.append(part)
        return parts

    def concat(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(parts, (list, tuple)) or len(parts) != len(self.factors):
            raise ValueError(f"ProductManifold.concat expects {len(self.factors)} parts, got {len(parts)}")
        # Validate shapes
        batch_shape = parts[0].shape[:-1]
        for p, f in zip(parts, self.factors):
            if not isinstance(p, torch.Tensor):
                raise TypeError("ProductManifold.concat expects torch.Tensor parts")
            if p.shape[:-1] != batch_shape:
                raise ValueError("ProductManifold.concat: all parts must share same batch shape")
            if p.shape[-1] != f.dim:
                raise ValueError(f"ProductManifold.concat: part for '{f.name}' expects dim={f.dim}, got {p.shape[-1]}")
        return torch.cat(list(parts), dim=-1)

    # ---------------------------
    # Core API
    # ---------------------------
    def proj(self, z: torch.Tensor) -> torch.Tensor:
        parts = self.split(z)
        out_parts = []
        for p, f in zip(parts, self.factors):
            out_parts.append(f.manifold.proj(p))
        return self.concat(out_parts)

    def proj_tan(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._check_same_shape(z, v, "proj_tan")
        z_parts = self.split(z)
        v_parts = self.split(v)
        out_parts = []
        for zp, vp, f in zip(z_parts, v_parts, self.factors):
            out_parts.append(f.manifold.proj_tan(zp, vp))
        return self.concat(out_parts)

    def exp_map(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._check_same_shape(z, v, "exp_map")
        z = self.proj(z)
        z_parts = self.split(z)
        v_parts = self.split(v)
        out_parts = []
        for zp, vp, f in zip(z_parts, v_parts, self.factors):
            out_parts.append(f.manifold.exp_map(zp, vp))
        return self.concat(out_parts)

    def log_map(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        self._check_same_shape(z, w, "log_map")
        z = self.proj(z)
        w = self.proj(w)
        z_parts = self.split(z)
        w_parts = self.split(w)
        out_parts = []
        for zp, wp, f in zip(z_parts, w_parts, self.factors):
            out_parts.append(f.manifold.log_map(zp, wp))
        return self.concat(out_parts)

    def squared_dist(self, z: torch.Tensor, w: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Product metric: sum of squared distances across factors.
        """
        self._check_same_shape(z, w, "squared_dist")
        z = self.proj(z)
        w = self.proj(w)
        z_parts = self.split(z)
        w_parts = self.split(w)

        total = None
        for zp, wp, f in zip(z_parts, w_parts, self.factors):
            man = f.manifold
            if hasattr(man, "squared_dist"):
                d2 = man.squared_dist(zp, wp, keepdim=True)
            else:
                d = man.dist(zp, wp, keepdim=True)
                d2 = d * d
            total = d2 if total is None else (total + d2)

        if total is None:
            raise RuntimeError("ProductManifold.squared_dist: no factors? (should never happen)")
        if not keepdim:
            total = total.squeeze(-1)
        return torch.clamp(total, min=0.0)

    def dist(self, z: torch.Tensor, w: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Product metric distance:
          d = sqrt( sum_k d_k^2 )
        """
        d2 = self.squared_dist(z, w, keepdim=True)
        d = torch.sqrt(torch.clamp(d2, min=self.eps))
        if not keepdim:
            d = d.squeeze(-1)
        return d

    def inner(self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Product metric inner product: sum_k <u_k, v_k>_{z_k}.
        """
        self._check_same_shape(u, v, "inner(u,v)")
        if not isinstance(z, torch.Tensor):
            raise TypeError("ProductManifold.inner expects torch.Tensor z")
        if z.shape[-1] != self.dim_total:
            raise ValueError(f"ProductManifold.inner expects last dim={self.dim_total}, got {z.shape[-1]}")

        z = self.proj(z)
        z_parts = self.split(z)
        u_parts = self.split(u)
        v_parts = self.split(v)

        total = None
        for zp, up, vp, f in zip(z_parts, u_parts, v_parts, self.factors):
            ip = f.manifold.inner(zp, up, vp, keepdim=True)
            total = ip if total is None else (total + ip)

        if total is None:
            raise RuntimeError("ProductManifold.inner: no factors? (should never happen)")
        if not keepdim:
            total = total.squeeze(-1)
        return total

    def norm(self, z: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Norm induced by product metric:
          ||v|| = sqrt( sum_k ||v_k||^2 )
        """
        if not isinstance(z, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("ProductManifold.norm expects torch.Tensor inputs")
        if z.shape != v.shape:
            raise ValueError(f"ProductManifold.norm shape mismatch: z{tuple(z.shape)} vs v{tuple(v.shape)}")
        if z.shape[-1] != self.dim_total:
            raise ValueError(f"ProductManifold.norm expects last dim={self.dim_total}, got {z.shape[-1]}")

        z = self.proj(z)
        z_parts = self.split(z)
        v_parts = self.split(v)

        total = None
        for zp, vp, f in zip(z_parts, v_parts, self.factors):
            n = f.manifold.norm(zp, vp, keepdim=True)
            n2 = n * n
            total = n2 if total is None else (total + n2)

        if total is None:
            raise RuntimeError("ProductManifold.norm: no factors? (should never happen)")
        out = torch.sqrt(torch.clamp(total, min=self.eps))
        if not keepdim:
            out = out.squeeze(-1)
        return out

    def random(
        self,
        *shape: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Sample random points independently from each factor and concatenate.

        shape describes the batch prefix; last dim is implied.
        Example: random(batch, ) returns tensor [batch, dim_total]
        """
        if len(shape) == 0:
            raise ValueError("ProductManifold.random requires a shape prefix, e.g. (batch,)")
        parts = []
        for f in self.factors:
            # factor manifold random should accept (*shape, dim_factor) OR (*shape, ...)
            # We attempt the common pattern random(*shape, dim)
            man = f.manifold
            try:
                p = man.random(*shape, f.dim, device=device, dtype=dtype)
            except TypeError:
                # fallback: sample without dim, then validate
                p = man.random(*shape, device=device, dtype=dtype)
            if p.shape[-1] != f.dim:
                raise ValueError(
                    f"ProductManifold.random: factor '{f.name}' returned last dim {p.shape[-1]}, expected {f.dim}"
                )
            parts.append(p)
        return self.concat(parts)

    # ---------------------------
    # Utilities
    # ---------------------------
    def _check_same_shape(self, a: torch.Tensor, b: torch.Tensor, where: str) -> None:
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"ProductManifold.{where} expects torch.Tensor inputs.")
        if a.shape != b.shape:
            raise ValueError(f"ProductManifold.{where} shape mismatch: a.shape={tuple(a.shape)} vs b.shape={tuple(b.shape)}")
        if a.shape[-1] != self.dim_total:
            raise ValueError(f"ProductManifold.{where} expects last dim={self.dim_total}, got {a.shape[-1]}")

