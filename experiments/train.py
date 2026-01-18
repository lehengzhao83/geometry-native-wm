# train.py
# Geometry-Native World Model training script (GUARANTEED TO RUN)
#
# Works with:
#   - configs/toy_hierarchy.yaml  -> datasets/toy_hierarchy.py
#   - configs/toy_periodic.yaml   -> datasets/toy_periodic.py
#   - configs/real_video.yaml     -> datasets/real_wrapper.py (falls back to FakeData pseudo-video)
#   - configs/toy_pose.yaml       -> datasets/toy_pose.py
#   - configs/vlm_binding.yaml    -> (can run as reconstruction/latent-pred; true VLM eval is separate)
#
# Minimal assumptions:
# - YAML provides:
#     experiment: {seed, device, dtype, out_dir, log_every, eval_every}
#     data: {name: toy_hierarchy | toy_periodic | real_video | toy_pose | vlm_binding, ...}
#     model: {encoder, latent, dynamics, decoder(optional)}
#     train: {batch_size, lr, weight_decay, epochs, grad_clip, num_workers}
#     loss: {use_recon: bool, use_latent: bool, latent_weight: float, recon_weight: float}
#
# If decoder is absent, it will train with latent-prediction loss:
#   L_latent = d_M(z_pred_next, z_true_next)^2
#
# If decoder exists, it will also train with reconstruction loss:
#   L_recon = MSE(x_hat_{t+1}, x_{t+1}) (masked if mask exists)
#
# Dependencies: torch, pyyaml, tqdm, tensorboard (optional)

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from manifolds.utils import seed_everything
from models.world_model import GeometryNativeWorldModel

from datasets.toy_hierarchy import build_toy_hierarchy_datasets
from datasets.toy_periodic import build_toy_periodic_datasets
from datasets.toy_pose import build_toy_pose_datasets
from datasets.real_wrapper import build_real_video_datasets


# -----------------------
# Config helpers
# -----------------------
from omegaconf import OmegaConf

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML config must parse to a dict.")

    # 关键：解析 ${...} 插值
    cfg = OmegaConf.create(raw)
    resolved = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(resolved, dict):
        raise ValueError(f"Resolved config must be dict, got {type(resolved)}")
    return resolved


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("experiment", {}).get("device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev in ("cpu", "cuda"):
        if dev == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(dev)
    raise ValueError(f"Unknown experiment.device: {dev}")


def maybe_make_tb(out_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    except Exception:
        return None


# -----------------------
# Dataset builders
# -----------------------
def build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("name", data_cfg.get("task", "toy_periodic"))).lower()

    if name in ("toy_hierarchy", "hierarchy"):
        return build_toy_hierarchy_datasets(cfg)

    if name in ("toy_periodic", "periodic", "torus"):
        return build_toy_periodic_datasets(cfg)

    if name in ("real_video", "video"):
        return build_real_video_datasets(cfg)

    if name in ("toy_pose", "pose"):
        return build_toy_pose_datasets(cfg)

    if name in ("vlm_binding", "binding"):
        # For now we treat it as a "real_video-like" sequence task by default.
        # If your vlm_binding.yaml points to real frame folders, it will load them;
        # otherwise it will fall back to FakeData pseudo-video and STILL RUN.
        return build_real_video_datasets(cfg)

    raise ValueError(f"Unknown data.name: {name}")


# -----------------------
# Batch parsing
# -----------------------
def extract_sequence(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      x:    [B, T, ...]
      mask: [B, T, D] or None
    We expect dataset returns:
      - "x": [T, ...] per sample -> collate => [B, T, ...]
      - optional "mask": [T, D]  -> collate => [B, T, D]
    """
    if "x" not in batch:
        raise KeyError("Batch must contain key 'x'")
    x = batch["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("batch['x'] must be a torch.Tensor")
    if x.dim() < 3:
        # For vector: [B,T,D] => dim=3
        # For image:  [B,T,C,H,W] => dim=5
        raise ValueError(f"batch['x'] must be sequence tensor with dim>=3, got shape {tuple(x.shape)}")

    mask = batch.get("mask", None)
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError("batch['mask'] must be torch.Tensor if provided")
    return x, mask


# -----------------------
# Losses
# -----------------------
def mse_recon_loss(x_hat: torch.Tensor, x_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    x_hat, x_true: same shape
    mask:
      - if vector: [B,T,D] applied elementwise
      - if image:  not used (unless provided as broadcastable)
    """
    if mask is None:
        return F.mse_loss(x_hat, x_true)

    # Broadcast mask if possible; otherwise require matching
    if mask.shape != x_true.shape:
        # try to broadcast: e.g., [B,T,1] to [B,T,D]
        try:
            m = mask
            while m.dim() < x_true.dim():
                m = m.unsqueeze(-1)
            return ((x_hat - x_true) ** 2 * m).sum() / (m.sum().clamp_min(1.0))
        except Exception:
            raise ValueError(f"Mask shape {tuple(mask.shape)} not compatible with x {tuple(x_true.shape)}")
    else:
        return ((x_hat - x_true) ** 2 * mask).sum() / (mask.sum().clamp_min(1.0))


def latent_pred_loss(world: GeometryNativeWorldModel, z_pred_next: torch.Tensor, z_true_next: torch.Tensor) -> torch.Tensor:
    # Use manifold distance squared
    # world.manifold.squared_dist should exist for Product/Circle; if not, fallback to dist^2.
    man = world.manifold
    if hasattr(man, "squared_dist"):
        d2 = man.squared_dist(z_pred_next, z_true_next, keepdim=False)
    else:
        d = man.dist(z_pred_next, z_true_next, keepdim=False)
        d2 = d * d
    return d2.mean()


# -----------------------
# Train / eval loops
# -----------------------
def run_epoch(
    world: GeometryNativeWorldModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: Dict[str, Any],
    train: bool,
) -> Dict[str, float]:
    world.train(train)

    loss_cfg = cfg.get("loss", {})
    use_recon = bool(loss_cfg.get("use_recon", True))
    use_latent = bool(loss_cfg.get("use_latent", True))
    w_recon = float(loss_cfg.get("recon_weight", 1.0))
    w_latent = float(loss_cfg.get("latent_weight", 1.0))

    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))

    total_loss = 0.0
    total_recon = 0.0
    total_latent = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=("train" if train else "eval"), leave=False)
    for batch in pbar:
        x, mask = extract_sequence(batch)
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        # We train one-step prediction across all t: x[:,0..T-2] -> predict x[:,1..T-1]
        # Prepare x_t and x_{t+1}
        x_t = x[:, :-1]
        x_tp1 = x[:, 1:]
        mask_tp1 = mask[:, 1:] if mask is not None and mask.dim() >= 3 else None

        # Forward world model
        out = world(x_t)  # outputs z,z_next,(x_next_hat optional)

        losses = []

        # (A) Reconstruction loss if decoder exists AND enabled
        recon = torch.tensor(0.0, device=device)
        if use_recon and out.x_next_hat is not None:
            # out.x_next_hat aligns with x_t leading dims, i.e. predicts x_{t+1}
            recon = mse_recon_loss(out.x_next_hat, x_tp1, mask_tp1)
            losses.append(w_recon * recon)

        # (B) Latent prediction loss (always possible)
        latent = torch.tensor(0.0, device=device)
        if use_latent:
            # Encode ground-truth next obs to z_true_next and compare on manifold
            with torch.set_grad_enabled(train):
                z_true_next = world.encode(x_tp1)
            latent = latent_pred_loss(world, out.z_next, z_true_next)
            losses.append(w_latent * latent)

        if len(losses) == 0:
            raise RuntimeError("No loss terms enabled. Set loss.use_recon/use_latent in yaml.")

        loss = torch.stack(losses).sum()

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(world.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_recon += float(recon.detach().cpu())
        total_latent += float(latent.detach().cpu())
        n_batches += 1

        pbar.set_postfix(
            loss=total_loss / max(1, n_batches),
            recon=total_recon / max(1, n_batches),
            latent=total_latent / max(1, n_batches),
        )

    return {
        "loss": total_loss / max(1, n_batches),
        "recon": total_recon / max(1, n_batches),
        "latent": total_latent / max(1, n_batches),
    }


def save_checkpoint(out_dir: str, world: GeometryNativeWorldModel, optimizer: torch.optim.Optimizer, epoch: int) -> str:
    ckpt = {
        "epoch": epoch,
        "model": world.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(out_dir, f"ckpt_epoch_{epoch:03d}.pt")
    torch.save(ckpt, path)
    return path


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/toy_periodic.yaml)")
    parser.add_argument("--override_out_dir", type=str, default=None, help="Override experiment.out_dir")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    exp_cfg = cfg.setdefault("experiment", {})
    train_cfg = cfg.setdefault("train", {})
    seed = int(exp_cfg.get("seed", 42))
    seed_everything(seed, deterministic=bool(exp_cfg.get("deterministic", True)))

    device = get_device(cfg)

    out_dir = str(args.override_out_dir or exp_cfg.get("out_dir", "./runs/default"))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "ckpts"))

    # TensorBoard (optional)
    tb = maybe_make_tb(out_dir)

    # Build datasets
    datasets = build_datasets(cfg)

    bs = int(train_cfg.get("batch_size", 64))
    nw = int(train_cfg.get("num_workers", 0))
    pin = bool(train_cfg.get("pin_memory", device.type == "cuda"))

    train_loader = DataLoader(datasets["train"], batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(datasets["val"], batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(datasets.get("test", datasets["val"]), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False)

    # Build model
    world = GeometryNativeWorldModel(cfg).to(device)

    # Optimizer
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(world.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 10))
    log_every = int(exp_cfg.get("log_every", 50))
    eval_every = int(exp_cfg.get("eval_every", 1))
    save_every = int(exp_cfg.get("save_every", 1))

    print(f"[train.py] device={device} out_dir={out_dir} epochs={epochs} batch_size={bs}")
    print(f"[train.py] data.name={cfg.get('data', {}).get('name', 'unknown')} geometry={cfg.get('model', {}).get('latent', {}).get('geometry', 'unknown')}")

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = run_epoch(world, train_loader, optimizer, device, cfg, train=True)

        # Eval
        if epoch % eval_every == 0:
            with torch.no_grad():
                val_metrics = run_epoch(world, val_loader, optimizer=None, device=device, cfg=cfg, train=False)
        else:
            val_metrics = {"loss": float("nan"), "recon": float("nan"), "latent": float("nan")}

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_metrics['loss']:.6f} (recon {train_metrics['recon']:.6f}, latent {train_metrics['latent']:.6f}) | "
            f"val loss {val_metrics['loss']:.6f} (recon {val_metrics['recon']:.6f}, latent {val_metrics['latent']:.6f}) | "
            f"{dt:.1f}s"
        )

        # TensorBoard
        if tb is not None:
            tb.add_scalar("train/loss", train_metrics["loss"], epoch)
            tb.add_scalar("train/recon", train_metrics["recon"], epoch)
            tb.add_scalar("train/latent", train_metrics["latent"], epoch)
            if epoch % eval_every == 0:
                tb.add_scalar("val/loss", val_metrics["loss"], epoch)
                tb.add_scalar("val/recon", val_metrics["recon"], epoch)
                tb.add_scalar("val/latent", val_metrics["latent"], epoch)

        # Save best + periodic checkpoints
        if epoch % eval_every == 0 and val_metrics["loss"] == val_metrics["loss"]:  # not nan
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = os.path.join(out_dir, "ckpts", "best.pt")
                torch.save({"epoch": epoch, "model": world.state_dict(), "optimizer": optimizer.state_dict()}, best_path)
                print(f"[train.py] saved best checkpoint -> {best_path}")

        if epoch % save_every == 0:
            path = save_checkpoint(os.path.join(out_dir, "ckpts"), world, optimizer, epoch)
            print(f"[train.py] saved checkpoint -> {path}")

        global_step += 1

    # Final test
    with torch.no_grad():
        test_metrics = run_epoch(world, test_loader, optimizer=None, device=device, cfg=cfg, train=False)
    print(
        f"[TEST] loss {test_metrics['loss']:.6f} (recon {test_metrics['recon']:.6f}, latent {test_metrics['latent']:.6f})"
    )

    if tb is not None:
        tb.close()


if __name__ == "__main__":
    main()
