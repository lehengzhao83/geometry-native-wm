# models/decoder.py
# "Hard to break" decoders used across toy + video (+ optional VLM image reconstruction).
#
# Supports:
#   - MLPDecoder: latent -> vector observation (toy_hierarchy / toy_periodic)
#   - SmallDeconvDecoder: latent -> image (real_video)
#
# Design goals:
# - torch-only
# - strict shape checks
# - predictable output shapes
#
# Expected config:
#   decoder.type: mlp          -> MLPDecoder
#   decoder.type: deconv_small -> SmallDeconvDecoder

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

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
    if name in ("elu",):
        return nn.ELU(inplace=True)
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


# -----------------------------
# Vector decoder (MLP)
# -----------------------------
class MLPDecoder(nn.Module):
    """
    Decodes latent vectors z[..., in_dim] -> x_hat[..., output_dim]
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
        ln = (lambda d: nn.LayerNorm(d)) if layernorm else (lambda d: nn.Identity())

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            if h <= 0:
                raise ValueError("hidden_dims must contain positive ints")
            layers.append(nn.Linear(prev, h))
            layers.append(ln(h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"MLPDecoder expects torch.Tensor, got {type(z)}")
        if z.shape[-1] != self.input_dim:
            raise ValueError(f"MLPDecoder: expected last dim {self.input_dim}, got {z.shape[-1]}")
        return self.net(z)


# -----------------------------
# Small Deconv decoder (latent -> image)
# -----------------------------
class SmallDeconvDecoder(nn.Module):
    """
    Decodes a latent vector to an image.

    Input:
      z: [B, in_dim] or [B, T, in_dim]
    Output:
      img: [B, C, H, W] or [B, T, C, H, W]

    Default output size: 64x64 if num_upsamples=4 (starting from 4x4).
    """

    def __init__(
        self,
        in_dim: int,
        out_channels: int = 3,
        base_channels: int = 64,
        num_upsamples: int = 4,
        activation: str = "gelu",
        out_activation: str = "sigmoid",  # sigmoid gives [0,1] images
    ):
        super().__init__()
        if not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be positive int, got {in_dim}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(f"out_channels must be positive int, got {out_channels}")
        if not isinstance(base_channels, int) or base_channels <= 0:
            raise ValueError(f"base_channels must be positive int, got {base_channels}")
        if not isinstance(num_upsamples, int) or num_upsamples < 2:
            raise ValueError("num_upsamples must be >= 2 (e.g., 4 for 64x64)")
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.num_upsamples = num_upsamples

        act = _act(activation)

        # Start from a small spatial grid: 4x4
        self.start_hw = 4
        # Choose channels progression
        ch = base_channels * (2 ** (num_upsamples - 1))  # e.g., 64*8=512 when num_upsamples=4
        self.fc = nn.Linear(in_dim, ch * self.start_hw * self.start_hw)

        blocks: List[nn.Module] = []
        cur_ch = ch
        for i in range(num_upsamples):
            # Upsample by factor 2 each step except last? We'll do num_upsamples-1 upsample to reach 64.
            is_last = (i == num_upsamples - 1)
            if not is_last:
                next_ch = max(base_channels, cur_ch // 2)
                blocks += [
                    nn.ConvTranspose2d(cur_ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_ch),
                    act,
                ]
                cur_ch = next_ch
            else:
                # final conv to output channels, keep spatial size
                blocks += [
                    nn.Conv2d(cur_ch, out_channels, kernel_size=3, stride=1, padding=1),
                ]

        self.net = nn.Sequential(*blocks)

        out_activation = str(out_activation).lower()
        if out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_activation == "tanh":
            self.out_act = nn.Tanh()
        elif out_activation in ("none", "identity"):
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {out_activation}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"SmallDeconvDecoder expects torch.Tensor, got {type(z)}")

        if z.dim() == 2:
            return self._forward_2d(z)

        if z.dim() == 3:
            b, t, d = z.shape
            if d != self.in_dim:
                raise ValueError(f"SmallDeconvDecoder: expected last dim {self.in_dim}, got {d}")
            z2 = z.reshape(b * t, d)
            img = self._forward_2d(z2).reshape(b, t, self.out_channels, -1, -1)
            # The reshape above can't infer H,W reliably; do it explicitly
            # We know output is square with size = 4 * 2^(num_upsamples-1)
            H = self.start_hw * (2 ** (self.num_upsamples - 1))
            W = H
            img = img.reshape(b, t, self.out_channels, H, W)
            return img

        raise ValueError(f"SmallDeconvDecoder: expected 2D or 3D input, got shape {tuple(z.shape)}")

    def _forward_2d(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.in_dim:
            raise ValueError(f"SmallDeconvDecoder: expected last dim {self.in_dim}, got {z.shape[-1]}")
        b = z.shape[0]
        y = self.fc(z)
        # reshape to [B, C, 4, 4]
        # infer C from fc out features
        C = y.shape[1] // (self.start_hw * self.start_hw)
        y = y.view(b, C, self.start_hw, self.start_hw)
        y = self.net(y)
        y = self.out_act(y)
        return y


# -----------------------------
# Factory
# -----------------------------
def build_decoder(cfg: dict) -> nn.Module:
    """
    Build a decoder from cfg.

    Expected patterns:
      {"type":"mlp", "input_dim":64, "hidden_dims":[256,256], "output_dim":64, ...}
      {"type":"deconv_small", "in_dim":80, "out_channels":3, ...}
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"build_decoder expects dict, got {type(cfg)}")
    if "type" not in cfg:
        raise ValueError("Decoder cfg must include 'type'")

    dtype = str(cfg["type"]).lower()

    if dtype == "mlp":
        return MLPDecoder(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=_ensure_list_int(cfg.get("hidden_dims", [256, 256]), "hidden_dims"),
            output_dim=int(cfg["output_dim"]),
            activation=str(cfg.get("activation", "gelu")),
            layernorm=bool(cfg.get("layernorm", True)),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    if dtype in ("deconv_small", "deconv"):
        # accept either "in_dim" or "input_dim"
        in_dim = int(cfg.get("in_dim", cfg.get("input_dim")))
        return SmallDeconvDecoder(
            in_dim=in_dim,
            out_channels=int(cfg.get("out_channels", 3)),
            base_channels=int(cfg.get("base_channels", 64)),
            num_upsamples=int(cfg.get("num_upsamples", 4)),
            activation=str(cfg.get("activation", "gelu")),
            out_activation=str(cfg.get("out_activation", "sigmoid")),
        )

    raise ValueError(f"Unknown decoder type: {dtype}")

