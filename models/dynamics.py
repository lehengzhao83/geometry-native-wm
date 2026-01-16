# models/dynamics.py
# "Hard to break" dynamics modules for Geometry-Native World Models.
#
# Supports:
#   - TangentMLPDynamics: outputs a tangent vector v_t given current state z_t (and optional action a_t)
#     then the caller applies exp_map(z_t, v_t) on the chosen manifold.
#
# Design goals:
# - minimal dependencies (torch only)
# - strict shape checks
# - compatible with Euclidean / Hyperbolic / Circle / Product manifolds
#
# Expected use:
#   v = dynamics(z)            # or dynamics(z, a)
#   z_next = manifold.exp_map(z, v)

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = str(name).lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def _ensure_list_int(x: Union[Sequence[int], int], name: str) -> List[int]:
    if isinstance(x, int):
        return [x]
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"{name} must be int or list/tuple of int, got {type(x)}")
    out = []
    for v in x:
        if not isinstance(v, int):
            raise TypeError(f"{name} entries must be int, got {type(v)}")
        out.append(v)
    if len(out) == 0:
        raise ValueError(f"{name} must be non-empty")
    return out


class TangentMLPDynamics(nn.Module):
    """
    MLP dynamics in tangent space.

    Inputs:
      z : [..., z_dim]
      a : [..., a_dim] (optional; if action_dim=0, ignored)

    Output:
      v : [..., z_dim]  (tangent vector to be mapped via exp_map)

    Notes:
    - This module does NOT apply exp_map itself.
    - You may optionally scale outputs for stability (step_scale).
    """

    def __init__(
        self,
        z_dim: int,
        action_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "gelu",
        layernorm: bool = True,
        dropout: float = 0.0,
        step_scale: float = 1.0,
    ):
        super().__init__()
        if not isinstance(z_dim, int) or z_dim <= 0:
            raise ValueError(f"z_dim must be positive int, got {z_dim}")
        if not isinstance(action_dim, int) or action_dim < 0:
            raise ValueError(f"action_dim must be int >= 0, got {action_dim}")
        hidden_dims = _ensure_list_int(hidden_dims, "hidden_dims")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")
        if not isinstance(step_scale, (int, float)) or step_scale <= 0:
            raise ValueError(f"step_scale must be > 0, got {step_scale}")

        self.z_dim = z_dim
        self.action_dim = action_dim
        self.step_scale = float(step_scale)

        act = _act(activation)
        ln = (lambda d: nn.LayerNorm(d)) if layernorm else (lambda d: nn.Identity())

        in_dim = z_dim + action_dim
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            if h <= 0:
                raise ValueError("hidden_dims must contain positive ints")
            layers.append(nn.Linear(prev, h))
            layers.append(ln(h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, z_dim))
        self.net = nn.Sequential(*layers)

        # Small init for final layer helps stability across manifolds
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, z: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"TangentMLPDynamics expects z as torch.Tensor, got {type(z)}")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"TangentMLPDynamics: expected z last dim {self.z_dim}, got {z.shape[-1]}")

        if self.action_dim == 0:
            x = z
        else:
            if a is None:
                raise ValueError("TangentMLPDynamics: action_dim > 0 but a is None")
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"TangentMLPDynamics expects a as torch.Tensor, got {type(a)}")
            if a.shape[:-1] != z.shape[:-1]:
                raise ValueError("TangentMLPDynamics: z and a must share the same batch shape")
            if a.shape[-1] != self.action_dim:
                raise ValueError(f"TangentMLPDynamics: expected a last dim {self.action_dim}, got {a.shape[-1]}")
            x = torch.cat([z, a], dim=-1)

        v = self.net(x)
        return v * self.step_scale


# -----------------------------
# Factory
# -----------------------------
def build_dynamics(cfg: dict, z_dim: int) -> nn.Module:
    """
    Build a dynamics module from cfg. Currently supports:
      type: "tangent_mlp"
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"build_dynamics expects dict, got {type(cfg)}")
    if "type" not in cfg:
        raise ValueError("Dynamics cfg must include 'type'")
    dtype = str(cfg["type"]).lower()

    if dtype in ("tangent_mlp", "mlp"):
        return TangentMLPDynamics(
            z_dim=int(z_dim),
            action_dim=int(cfg.get("action_dim", 0)),
            hidden_dims=_ensure_list_int(cfg.get("hidden_dims", [256, 256]), "hidden_dims"),
            activation=str(cfg.get("activation", "gelu")),
            layernorm=bool(cfg.get("layernorm", True)),
            dropout=float(cfg.get("dropout", 0.0)),
            step_scale=float(cfg.get("step_scale", 1.0)),
        )

    raise ValueError(f"Unknown dynamics type: {dtype}")

