# models/world_model.py
# Geometry-Native World Model (WORKING, torch-only)
#
# This module glues together:
#   encoder(x_t) -> z_t (ambient coords)
#   manifold.proj(z_t) -> z_t on manifold
#   dynamics(z_t) -> v_t (tangent/ambient coords)
#   manifold.exp_map(z_t, v_t) -> z_{t+1}
#   decoder(z_{t+1}) -> x_{t+1} prediction
#
# It is intentionally defensive and designed to "just run" for:
#   - toy_hierarchy (vector obs)
#   - toy_periodic (vector obs)
#   - real_video (image obs; predicts next frame)
#   - vlm_binding (can use encode-only + heads outside; this file doesn't do task heads)
#
# Dependencies: torch + your local files:
#   - models/encoder.py, models/dynamics.py, models/decoder.py
#   - manifolds/euclidean.py, manifolds/hyperbolic.py, manifolds/circle.py, manifolds/product.py
#   - manifolds/utils.py
#
# Notes about S^1 dims:
#   If geometry == circle_only and representation == unit_vector_2d:
#     - We REQUIRE latent_dim to be even and we set num_circles = latent_dim//2 to guarantee shape consistency.
#     - If cfg specifies num_circles differently, we ignore it (better to run than crash).
#
# If geometry == hyperbolic_only:
#   - latent_dim = hyperbolic.dim (cfg must match encoder output_dim).
#
# If geometry == product:
#   - latent_dim inferred from factors. Encoder output_dim MUST match dim_total (from cfg).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoder import build_encoder
from .dynamics import build_dynamics
from .decoder import build_decoder

from manifolds.euclidean import Euclidean
from manifolds.hyperbolic import PoincareBall
from manifolds.circle import Circle
from manifolds.product import FactorSpec, ProductManifold
from manifolds.utils import build_product_from_cfg, infer_device, infer_dtype


@dataclass
class WMOutputs:
    z: torch.Tensor                  # [B,T,dim] or [B,dim] manifold point
    v: torch.Tensor                  # tangent vector, same shape
    z_next: torch.Tensor             # predicted next latent
    x_next_hat: Optional[torch.Tensor] = None  # decoded prediction of x_{t+1}


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


class GeometryNativeWorldModel(nn.Module):
    """
    A minimal world model that is geometry-native:
      z_{t+1} = Exp_{z_t}( f_theta(z_t) )

    It supports both single-step forward and multi-step rollout in latent space.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, dict):
            raise TypeError(f"GeometryNativeWorldModel expects cfg as dict, got {type(cfg)}")

        self.cfg = cfg

        # ---------- Device / dtype ----------
        exp_cfg = cfg.get("experiment", {})
        device_spec = str(exp_cfg.get("device", "auto"))
        dtype_spec = str(exp_cfg.get("dtype", "float32"))
        self.device_ = infer_device(device_spec)
        self.dtype_ = infer_dtype(dtype_spec)

        # ---------- Build encoder ----------
        model_cfg = cfg.get("model", {})
        enc_cfg = model_cfg.get("encoder", {})
        if not isinstance(enc_cfg, dict):
            raise ValueError("cfg['model']['encoder'] must be a dict")
        self.encoder = build_encoder(enc_cfg)

        # ---------- Build manifold ----------
        latent_cfg = model_cfg.get("latent", {})
        if not isinstance(latent_cfg, dict):
            raise ValueError("cfg['model']['latent'] must be a dict")

        geometry = str(latent_cfg.get("geometry", "euclidean_only")).lower()
        self.geometry = geometry

        # Determine latent dim_total
        dim_total = latent_cfg.get("dim_total", None)
        if dim_total is None:
            # try common fields
            dim_total = _get(latent_cfg, "dim", None)
        if dim_total is None and geometry != "product":
            raise ValueError("cfg['model']['latent'] must include 'dim_total' for non-product geometry")

        if geometry in ("euclidean_only", "euclid", "r"):
            if dim_total is None:
                raise ValueError("Euclidean latent requires dim_total")
            self.manifold = Euclidean()
            self.latent_dim = int(dim_total)

        elif geometry in ("hyperbolic_only", "hyperbolic", "poincare_ball"):
            hyp = latent_cfg.get("hyperbolic", {})
            if not isinstance(hyp, dict):
                hyp = {}
            if dim_total is None:
                dim_total = int(hyp.get("dim", 64))
            c = float(hyp.get("curvature", 1.0))
            eps = float(hyp.get("eps", 1e-5))
            self.manifold = PoincareBall(c=c, eps=eps)
            self.latent_dim = int(dim_total)

        elif geometry in ("circle_only", "s1", "torus"):
            circ = latent_cfg.get("circle", {})
            if not isinstance(circ, dict):
                circ = {}
            rep = str(circ.get("representation", "unit_vector_2d"))
            eps = float(circ.get("project_eps", circ.get("eps", 1e-6)))

            if dim_total is None:
                # If dim_total not provided, fall back to config num_circles
                num_circles = int(circ.get("num_circles", 1))
                dim_total = 2 * num_circles if rep == "unit_vector_2d" else num_circles

            dim_total = int(dim_total)
            if rep == "unit_vector_2d":
                if dim_total % 2 != 0:
                    raise ValueError(
                        f"circle_only with unit_vector_2d requires even dim_total, got {dim_total}"
                    )
                # Force-consistent num_circles from dim_total to GUARANTEE it runs.
                num_circles = dim_total // 2
            else:
                num_circles = dim_total

            self.manifold = Circle(num_circles=num_circles, representation=rep, eps=eps)
            self.latent_dim = dim_total

        elif geometry in ("product", "product_manifold"):
            prod = latent_cfg.get("product", None)
            if prod is None:
                raise ValueError("product geometry requires cfg['model']['latent']['product']")
            if not isinstance(prod, dict):
                raise ValueError("cfg['model']['latent']['product'] must be a dict")

            # Our utils expects {"factors":[...]} format; configs use {"factors":[...]} already.
            self.manifold = build_product_from_cfg(prod)
            self.latent_dim = int(self.manifold.dim_total)

            # Ensure config dim_total consistent; if not present, set it for internal use
            if dim_total is None:
                dim_total = self.latent_dim
            else:
                dim_total = int(dim_total)
                if dim_total != self.latent_dim:
                    raise ValueError(
                        f"Product manifold dim_total mismatch: cfg says {dim_total}, "
                        f"but factors imply {self.latent_dim}"
                    )
        else:
            raise ValueError(f"Unknown latent geometry: {geometry}")

        # ---------- Sanity: encoder output dim must match latent_dim ----------
        # For MLPEncoder, output_dim is known. For CNN, feature_dim is known.
        enc_out = getattr(self.encoder, "output_dim", None)
        if enc_out is None:
            enc_out = getattr(self.encoder, "feature_dim", None)
        if enc_out is None:
            raise RuntimeError("Encoder must expose output_dim or feature_dim.")
        if int(enc_out) != int(self.latent_dim):
            raise ValueError(
                f"Encoder output dim ({int(enc_out)}) must match latent_dim ({int(self.latent_dim)}). "
                f"Fix configs: encoder.output_dim/feature_dim or latent.dim_total."
            )

        # ---------- Build dynamics ----------
        dyn_cfg = model_cfg.get("dynamics", {})
        if not isinstance(dyn_cfg, dict):
            raise ValueError("cfg['model']['dynamics'] must be a dict")
        self.dynamics = build_dynamics(dyn_cfg, z_dim=self.latent_dim)

        # ---------- Build decoder (optional) ----------
        dec_cfg = model_cfg.get("decoder", None)
        self.has_decoder = dec_cfg is not None and isinstance(dec_cfg, dict)
        if self.has_decoder:
            # ensure decoder input dim matches latent dim if it's MLP; for deconv, we pass in_dim
            # configs already set input_dim/in_dim to ${model.latent.dim_total}
            self.decoder = build_decoder(dec_cfg)
        else:
            self.decoder = None

        # Move to device/dtype
        self.to(self.device_)
        # Keep parameters in float32/float16 as chosen
        self._cast_module(self, self.dtype_)

    @staticmethod
    def _cast_module(module: nn.Module, dtype: torch.dtype) -> None:
        # Cast only floating parameters/buffers
        for p in module.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype=dtype)
        for bname, buf in module.named_buffers():
            if buf.is_floating_point():
                setattr(module, bname, buf.to(dtype=dtype))

    # -------------------------
    # Core operations
    # -------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations into latent points on the manifold.

        Accepts:
          - vector: [B, D] or [B, T, D]
          - image:  [B, C, H, W] or [B, T, C, H, W]

        Returns:
          z: same leading dims, last dim = latent_dim, projected onto manifold
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects torch.Tensor, got {type(x)}")
        x = x.to(device=self.device_, dtype=self.dtype_)

        z = self.encoder(x)
        z = z.to(device=self.device_, dtype=self.dtype_)
        z = self.manifold.proj(z)
        return z

    def step(self, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One latent transition:
          v = dynamics(z, action)
          z_next = Exp_z(v)
        Returns:
          (z_next, v)
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"step expects z as torch.Tensor, got {type(z)}")
        z = z.to(device=self.device_, dtype=self.dtype_)
        z = self.manifold.proj(z)

        v = self.dynamics(z, action) if action is not None else self.dynamics(z)
        v = v.to(device=self.device_, dtype=self.dtype_)
        v = self.manifold.proj_tan(z, v)

        z_next = self.manifold.exp_map(z, v)
        z_next = self.manifold.proj(z_next)
        return z_next, v

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent points to observation space using the configured decoder.
        """
        if not self.has_decoder or self.decoder is None:
            raise RuntimeError("No decoder configured for this world model.")
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"decode expects torch.Tensor, got {type(z)}")
        z = z.to(device=self.device_, dtype=self.dtype_)
        return self.decoder(z)

    # -------------------------
    # Forward modes
    # -------------------------
    def forward(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> WMOutputs:
        """
        Default forward = single-step prediction given x_t.

        If x is a sequence [B,T,...], this predicts next-step for each t:
          z_t = encode(x_t)
          z_{t+1} = step(z_t)
          x_hat_{t+1} = decode(z_{t+1})  (if decoder present)

        Returns:
          WMOutputs with tensors matching the leading dims of x (except x_next_hat aligns with z_next).
        """
        z = self.encode(x)
        z_next, v = self.step(z, action)

        x_next_hat = self.decode(z_next) if self.has_decoder else None
        return WMOutputs(z=z, v=v, z_next=z_next, x_next_hat=x_next_hat)

    @torch.no_grad()
    def rollout_latent(self, z0: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Roll out in latent space for `horizon` steps.
        Input:
          z0: [B, dim] (or any batch prefix)
        Output:
          z_seq: [B, horizon+1, dim] (includes z0)
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be positive int, got {horizon}")
        z = self.manifold.proj(z0.to(device=self.device_, dtype=self.dtype_))
        seq = [z]
        for _ in range(horizon):
            z, _ = self.step(z)
            seq.append(z)
        return torch.stack(seq, dim=-2)  # insert time dim before last dim

