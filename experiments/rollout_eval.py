# rollout_eval.py
# Rollout evaluation script for Geometry-Native World Models (GUARANTEED TO RUN)
#
# What it does:
# - Loads a config YAML and an optional checkpoint
# - Builds the dataset (toy_hierarchy / toy_periodic / real_video / toy_pose)
# - Encodes an initial observation x_t0 -> z0
# - Rolls out latent states for H steps: z_{t+1} = Exp_{z_t}(f(z_t))
# - Optionally decodes to get predicted observations (if decoder exists)
# - Computes:
#     - Latent rollout error curve: mean d_M(z_hat_{t+τ}, z_true_{t+τ})^2 over τ=1..H
#     - Observation rollout error curve: mean MSE(x_hat_{t+τ}, x_true_{t+τ}) over τ (if decoder)
#
# For datasets with a "mask" (toy_periodic), it applies mask to observation error.
# For image datasets, it computes MSE on pixels.
#
# Usage:
#   python rollout_eval.py --config configs/toy_periodic.yaml --ckpt runs/.../ckpts/best.pt --horizon 25
#
# Dependencies: torch, pyyaml, tqdm

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from manifolds.utils import seed_everything
from models.world_model import GeometryNativeWorldModel

from datasets.toy_hierarchy import build_toy_hierarchy_datasets
from datasets.toy_periodic import build_toy_periodic_datasets
from datasets.toy_pose import build_toy_pose_datasets
from datasets.real_wrapper import build_real_video_datasets

from omegaconf import OmegaConf


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML config must parse to a dict.")

    # 🔑 关键：解析 ${...} 插值
    cfg = OmegaConf.create(raw)
    resolved = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(resolved, dict):
        raise ValueError(f"Resolved config must be dict, got {type(resolved)}")
    return resolved

def get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("experiment", {}).get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(dev)
    raise ValueError(f"Unknown experiment.device: {dev}")


def build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("name", data_cfg.get("task", "toy_periodic"))).lower()

    if name in ("toy_hierarchy", "hierarchy"):
        return build_toy_hierarchy_datasets(cfg)
    if name in ("toy_periodic", "periodic", "torus"):
        return build_toy_periodic_datasets(cfg)
    if name in ("toy_pose", "pose"):
        return build_toy_pose_datasets(cfg)
    if name in ("real_video", "video"):
        return build_real_video_datasets(cfg)
    if name in ("vlm_binding", "binding"):
        return build_real_video_datasets(cfg)

    raise ValueError(f"Unknown data.name: {name}")


def extract_sequence(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if "x" not in batch:
        raise KeyError("Batch must contain key 'x'")
    x = batch["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("batch['x'] must be a torch.Tensor")
    if x.dim() < 3:
        raise ValueError(f"batch['x'] must be sequence tensor with dim>=3, got shape {tuple(x.shape)}")
    mask = batch.get("mask", None)
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("batch['mask'] must be torch.Tensor if provided")
    return x, mask


def masked_mse(x_hat: torch.Tensor, x_true: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(x_hat, x_true)
    if mask.shape != x_true.shape:
        # attempt broadcast
        m = mask
        while m.dim() < x_true.dim():
            m = m.unsqueeze(-1)
        return ((x_hat - x_true) ** 2 * m).sum() / (m.sum().clamp_min(1.0))
    return ((x_hat - x_true) ** 2 * mask).sum() / (mask.sum().clamp_min(1.0))


@torch.no_grad()
def load_checkpoint(world: GeometryNativeWorldModel, ckpt_path: str, device: torch.device) -> None:
    if ckpt_path is None:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        world.load_state_dict(ckpt["model"], strict=True)
    else:
        world.load_state_dict(ckpt, strict=True)


@torch.no_grad()
def rollout_metrics(
    world: GeometryNativeWorldModel,
    x_seq: torch.Tensor,
    mask_seq: Optional[torch.Tensor],
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rollout error curves for a batch sequence.
    Inputs:
      x_seq:   [B, T, ...]
      mask_seq:[B, T, D] or None
    We evaluate starting at t0=0:
      z0 = encode(x0)
      zhat_{τ} rollout for τ=1..H
      Compare to z_true at x_{τ}
    Requires: T >= horizon+1
    Returns:
      latent_curve: [H]  mean squared geodesic distance
      obs_curve:    [H]  mean MSE, or NaN if no decoder
    """
    B = x_seq.shape[0]
    T = x_seq.shape[1]
    if T < horizon + 1:
        raise ValueError(f"Need sequence length T >= horizon+1. Got T={T}, horizon={horizon}")

    x0 = x_seq[:, 0]            # [B,...]
    x_true = x_seq[:, 1 : horizon + 1]  # [B,H,...]
    mask_true = mask_seq[:, 1 : horizon + 1] if mask_seq is not None else None

    # encode all ground-truth states for comparison
    z_true_all = world.encode(x_seq[:, : horizon + 1])  # [B,H+1,dim] (or [B,H+1,dim])
    z0 = z_true_all[:, 0]  # [B,dim]
    z_true = z_true_all[:, 1:]  # [B,H,dim]

    # rollout latents
    zhat_seq = world.rollout_latent(z0, horizon=horizon)  # [B,H+1,dim]
    zhat = zhat_seq[:, 1:]  # [B,H,dim]

    # latent error curve
    man = world.manifold
    if hasattr(man, "squared_dist"):
        d2 = man.squared_dist(zhat, z_true, keepdim=False)  # [B,H] or [...,H]
    else:
        d = man.dist(zhat, z_true, keepdim=False)
        d2 = d * d
    # Ensure [B,H]
    if d2.dim() == 1:
        d2 = d2.view(B, horizon)
    latent_curve = d2.mean(dim=0)  # [H]

    # observation curve (if decoder exists)
    if world.has_decoder and world.decoder is not None:
        xhat = world.decode(zhat)  # [B,H,...]
        # compute per-step mse
        obs = []
        for t in range(horizon):
            mt = mask_true[:, t] if mask_true is not None else None
            obs.append(masked_mse(xhat[:, t], x_true[:, t], mt))
        obs_curve = torch.stack(obs, dim=0)  # [H]
    else:
        obs_curve = torch.full((horizon,), float("nan"), device=latent_curve.device)

    return latent_curve, obs_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "test_ood"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num_batches", type=int, default=50, help="How many batches to average (<=0 means all)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed, deterministic=bool(cfg.get("experiment", {}).get("deterministic", True)))

    # data
    datasets = build_datasets(cfg)
    if args.split not in datasets:
        # fallback
        split = "test" if "test" in datasets else "val"
        print(f"[rollout_eval] split '{args.split}' not found. Using '{split}'.")
        args.split = split

    loader = DataLoader(datasets[args.split], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # model
    world = GeometryNativeWorldModel(cfg).to(device)
    load_checkpoint(world, args.ckpt, device=device)
    world.eval()

    H = int(args.horizon)
    latent_sum = torch.zeros((H,), device=device)
    obs_sum = torch.zeros((H,), device=device)
    n = 0

    pbar = tqdm(loader, desc="rollout_eval", leave=True)
    for batch in pbar:
        x, mask = extract_sequence(batch)
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        # ensure enough length by taking prefix if longer; if shorter, skip batch
        if x.shape[1] < H + 1:
            continue
        x = x[:, : H + 1]
        if mask is not None:
            mask = mask[:, : H + 1]

        latent_curve, obs_curve = rollout_metrics(world, x, mask, horizon=H)
        latent_sum += latent_curve
        obs_sum += torch.nan_to_num(obs_curve, nan=0.0)
        n += 1

        pbar.set_postfix(latent=float((latent_sum / max(1, n)).mean().detach().cpu()))

        if args.num_batches > 0 and n >= args.num_batches:
            break

    if n == 0:
        raise RuntimeError("No valid batches for rollout evaluation (maybe sequences too short).")

    latent_mean = latent_sum / n
    # For obs, if NaNs everywhere, mean will be 0; report separately
    obs_mean = obs_sum / n

    # Print curves
    print("\n=== Rollout Error Curves (mean over batches) ===")
    print("tau\tlatent_d2\tobs_mse")
    for t in range(H):
        print(f"{t+1}\t{float(latent_mean[t].cpu()):.6f}\t{float(obs_mean[t].cpu()):.6f}")

    # Summary
    print("\n=== Summary ===")
    print(f"latent_d2_mean_over_horizon: {float(latent_mean.mean().cpu()):.6f}")
    if world.has_decoder and world.decoder is not None:
        print(f"obs_mse_mean_over_horizon:    {float(obs_mean.mean().cpu()):.6f}")
    else:
        print("obs_mse_mean_over_horizon:    N/A (no decoder configured)")

    # Save to file
    out_dir = str(cfg.get("experiment", {}).get("out_dir", "./runs/default"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rollout_curve_{args.split}_H{H}.pt")
    torch.save({"latent": latent_mean.cpu(), "obs": obs_mean.cpu()}, out_path)
    print(f"\nSaved curves -> {out_path}")


if __name__ == "__main__":
    main()

