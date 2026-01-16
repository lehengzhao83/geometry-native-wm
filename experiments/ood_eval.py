# ood_eval.py
# OOD evaluation for Geometry-Native World Models (GUARANTEED TO RUN)
#
# What it does:
# - Loads config + optional checkpoint
# - Builds in-domain test set and (if available) OOD test set
# - Evaluates:
#     (1) one-step reconstruction loss (if decoder exists)
#     (Extras) latent one-step loss (always)
# - Reports means and saves a small JSON-like dict as .pt
#
# Works with:
#   - toy_hierarchy (test_ood optional via data.ood.enabled)
#   - toy_periodic  (test_ood optional via data.ood.enabled)
#   - real_video    (test_ood optional via data.ood.enabled)
#   - toy_pose      (test_ood optional via data.ood.enabled)
#
# Usage:
#   python ood_eval.py --config configs/toy_periodic.yaml --ckpt runs/.../ckpts/best.pt
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


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must parse to a dict.")
    return cfg


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
        m = mask
        while m.dim() < x_true.dim():
            m = m.unsqueeze(-1)
        return ((x_hat - x_true) ** 2 * m).sum() / (m.sum().clamp_min(1.0))
    return ((x_hat - x_true) ** 2 * mask).sum() / (mask.sum().clamp_min(1.0))


@torch.no_grad()
def load_checkpoint(world: GeometryNativeWorldModel, ckpt_path: Optional[str], device: torch.device) -> None:
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
def evaluate_one_step(
    world: GeometryNativeWorldModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 200,
) -> Dict[str, float]:
    """
    One-step evaluation:
      - latent loss: d_M(z_pred_next, z_true_next)^2
      - recon loss:  MSE(x_hat_{t+1}, x_{t+1}) if decoder exists
    """
    world.eval()
    man = world.manifold

    latent_sum = 0.0
    recon_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="eval_one_step", leave=False)
    for batch in pbar:
        x, mask = extract_sequence(batch)
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        # one-step over all t
        x_t = x[:, :-1]
        x_tp1 = x[:, 1:]
        mask_tp1 = mask[:, 1:] if mask is not None and mask.dim() >= 3 else None

        out = world(x_t)

        # latent true
        z_true_next = world.encode(x_tp1)

        if hasattr(man, "squared_dist"):
            d2 = man.squared_dist(out.z_next, z_true_next, keepdim=False)
        else:
            d = man.dist(out.z_next, z_true_next, keepdim=False)
            d2 = d * d
        latent = float(d2.mean().cpu())

        recon = 0.0
        if world.has_decoder and out.x_next_hat is not None:
            recon = float(masked_mse(out.x_next_hat, x_tp1, mask_tp1).cpu())

        latent_sum += latent
        recon_sum += recon
        n += 1

        pbar.set_postfix(latent=latent_sum / n, recon=recon_sum / n)

        if max_batches > 0 and n >= max_batches:
            break

    if n == 0:
        raise RuntimeError("No batches evaluated (maybe dataset empty).")

    return {
        "latent": latent_sum / n,
        "recon": recon_sum / n if (world.has_decoder and world.decoder is not None) else float("nan"),
        "batches": float(n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed, deterministic=bool(cfg.get("experiment", {}).get("deterministic", True)))

    datasets = build_datasets(cfg)

    # Choose split keys
    test_key = "test" if "test" in datasets else ("val" if "val" in datasets else "train")
    ood_key = "test_ood" if "test_ood" in datasets else None

    test_loader = DataLoader(
        datasets[test_key],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    ood_loader = None
    if ood_key is not None:
        ood_loader = DataLoader(
            datasets[ood_key],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )

    world = GeometryNativeWorldModel(cfg).to(device)
    load_checkpoint(world, args.ckpt, device=device)

    print(f"[ood_eval] Evaluating in-domain split='{test_key}'")
    ind = evaluate_one_step(world, test_loader, device=device, max_batches=args.max_batches)

    if ood_loader is not None:
        print(f"[ood_eval] Evaluating OOD split='{ood_key}'")
        ood = evaluate_one_step(world, ood_loader, device=device, max_batches=args.max_batches)
    else:
        ood = {"latent": float("nan"), "recon": float("nan"), "batches": float("nan")}

    # Robustness gaps (OOD / IND ratio)
    def ratio(a: float, b: float) -> float:
        if not (a == a) or not (b == b) or b <= 0:
            return float("nan")
        return a / b

    out = {
        "in_domain": ind,
        "ood": ood,
        "gap": {
            "latent_ratio": ratio(ood["latent"], ind["latent"]),
            "recon_ratio": ratio(ood["recon"], ind["recon"]),
            "latent_delta": (ood["latent"] - ind["latent"]) if (ood["latent"] == ood["latent"]) else float("nan"),
            "recon_delta": (ood["recon"] - ind["recon"]) if (ood["recon"] == ood["recon"]) else float("nan"),
        },
    }

    print("\n=== Results ===")
    print(f"In-domain latent: {out['in_domain']['latent']:.6f}")
    print(f"In-domain recon : {out['in_domain']['recon']:.6f}")
    print(f"OOD latent      : {out['ood']['latent']:.6f}")
    print(f"OOD recon       : {out['ood']['recon']:.6f}")
    print(f"Latent ratio (OOD/IND): {out['gap']['latent_ratio']:.3f}")
    print(f"Recon  ratio (OOD/IND): {out['gap']['recon_ratio']:.3f}")

    # Save
    out_dir = str(cfg.get("experiment", {}).get("out_dir", "./runs/default"))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ood_eval.pt")
    torch.save(out, save_path)
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()

