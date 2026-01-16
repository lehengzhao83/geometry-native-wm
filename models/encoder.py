# models/encoder.py
# "Hard to break" encoders used across toy + video + VLM configs.
#
# Supports:
#   - MLPEncoder: for vector observations (toy_hierarchy / toy_periodic)
#   - SmallCNNEncoder: for images (real_video / vlm_binding)
#
# Design goals:
#   - Minimal dependencies (torch only)
#   - Defensive shape checks
#   - Clear output_dim / feature_dim attributes
#
# Usage patterns (expected by your configs):
#   encoder.type: mlp        -> MLPEncoder
#   encoder.type: cnn_small  -> SmallCNNEncoder

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers
# -----------------------------
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
    if name in ("elu",):
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _maybe_layernorm(dim: int, enabled: bool) -> nn.Module:
    return nn.LayerNorm(dim) if enabled else nn.Identity()


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


# -----------------------------
# MLP Encoder (vector inputs)
# -----------------------------
class MLPEncoder(nn.Module):
    """
    Encodes vector observations x[..., input_dim] -> features[..., output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "gelu",
        layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be positive int, got {input_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be positive int, got {output_dim}")
        hidden_dims = _ensure_list_int(hidden_dims, "hidden_dims")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.input_dim = input_dim
        self.output_dim = output_dim

        act = _act(activation)
        layers: List[nn.Module] = []

        prev = input_dim
        for h in hidden_dims:
            if h <= 0:
                raise ValueError("hidden_dims must contain positive ints")
            layers.append(nn.Linear(prev, h))
            layers.append(_maybe_layernorm(h, layernorm))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"MLPEncoder expects torch.Tensor, got {type(x)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"MLPEncoder: expected last dim {self.input_dim}, got {x.shape[-1]}")
        return self.net(x)


# -----------------------------
# Small CNN Encoder (image inputs)
# -----------------------------
class SmallCNNEncoder(nn.Module):
    """
    Encodes images into a feature vector.

    Accepts:
      - [B, C, H, W]
      - [B, T, C, H, W]  -> returns [B, T, feature_dim] (frame-wise encoding)

    Output:
      - feature vectors (not logits). Use downstream modules for further mapping.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        feature_dim: int = 256,
        layernorm: bool = False,
    ):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels must be positive int, got {in_channels}")
        if not isinstance(base_channels, int) or base_channels <= 0:
            raise ValueError(f"base_channels must be positive int, got {base_channels}")
        if not isinstance(num_layers, int) or num_layers < 2:
            raise ValueError("num_layers must be >= 2 for a useful CNN")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive int, got {feature_dim}")

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        act = _act(activation)

        # Stem
        layers: List[nn.Module] = []
        ch = base_channels
        layers += [
            nn.Conv2d(in_channels, ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            act,
        ]

        # Downsample blocks
        for i in range(1, num_layers):
            ch_next = ch * 2 if i < num_layers - 1 else ch
            stride = 2 if i < num_layers - 1 else 1
            layers += [
                nn.Conv2d(ch, ch_next, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(ch_next),
                act,
            ]
            ch = ch_next

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(ch, feature_dim)
        self.ln = nn.LayerNorm(feature_dim) if layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"SmallCNNEncoder expects torch.Tensor, got {type(x)}")

        if x.dim() == 4:
            # [B, C, H, W]
            return self._forward_4d(x)

        if x.dim() == 5:
            # [B, T, C, H, W] -> [B, T, F]
            b, t, c, h, w = x.shape
            if c != self.in_channels:
                raise ValueError(f"SmallCNNEncoder: expected C={self.in_channels}, got {c}")
            x2 = x.reshape(b * t, c, h, w)
            f = self._forward_4d(x2).reshape(b, t, -1)
            return f

        raise ValueError(f"SmallCNNEncoder: expected 4D or 5D input, got shape {tuple(x.shape)}")

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("internal: _forward_4d expects 4D input")
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"SmallCNNEncoder: expected C={self.in_channels}, got {c}")

        y = self.conv(x)
        y = self.pool(y).flatten(1)  # [B, ch]
        y = self.proj(y)             # [B, feature_dim]
        y = self.ln(y)
        return y


# -----------------------------
# Factory (recommended)
# -----------------------------
def build_encoder(cfg: dict) -> nn.Module:
    """
    Build an encoder from a config dict.

    Expected cfg patterns:
      {"type":"mlp", "input_dim":64, "hidden_dims":[256,256], "output_dim":64, ...}
      {"type":"cnn_small", "in_channels":3, "base_channels":64, "num_layers":4, "feature_dim":256, ...}
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"build_encoder expects dict, got {type(cfg)}")
    if "type" not in cfg:
        raise ValueError("Encoder cfg must include 'type'")

    etype = str(cfg["type"]).lower()

    if etype == "mlp":
        return MLPEncoder(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=_ensure_list_int(cfg.get("hidden_dims", [256, 256]), "hidden_dims"),
            output_dim=int(cfg["output_dim"]),
            activation=str(cfg.get("activation", "gelu")),
            layernorm=bool(cfg.get("layernorm", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    if etype in ("cnn_small", "cnn"):
        return SmallCNNEncoder(
            in_channels=int(cfg.get("in_channels", 3)),
            base_channels=int(cfg.get("base_channels", 64)),
            num_layers=int(cfg.get("num_layers", 4)),
            activation=str(cfg.get("activation", "gelu")),
            feature_dim=int(cfg.get("feature_dim", cfg.get("output_dim", 256))),
            layernorm=bool(cfg.get("layernorm", False)),
        )

    raise ValueError(f"Unknown encoder type: {etype}")

